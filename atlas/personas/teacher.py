"""Teacher responsible for plan review, validation, and guidance."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple
from typing import Literal
from typing import cast

from atlas.config.models import TeacherConfig
from atlas.prompts import RewrittenTeacherPrompts
from atlas.types import Plan
from atlas.types import Step
from atlas.utils.llm_client import LLMClient
from atlas.runtime.orchestration.execution_context import ExecutionContext


class Teacher:
    def __init__(self, config: TeacherConfig, prompts: RewrittenTeacherPrompts) -> None:
        self._config = config
        self._client = LLMClient(config.llm)
        self._plan_cache: Dict[str, Tuple[float, Plan]] = {}
        self._plan_prompt = prompts.plan_review
        self._validation_prompt = prompts.validation
        self._guidance_prompt = prompts.guidance

    async def areview_plan(self, task: str, plan: Plan) -> Plan:
        cache_key = self._cache_key(task, plan)
        now = time.time()
        cached = self._plan_cache.get(cache_key)
        if cached and now - cached[0] <= self._config.plan_cache_seconds:
            return cached[1]
        context = ExecutionContext.get()
        context.metadata["active_actor"] = "teacher"
        context.metadata["_reasoning_origin"] = ("teacher", "plan_review")
        messages = [
            {"role": "system", "content": self._plan_prompt},
            {
                "role": "user",
                "content": json.dumps({"task": task, "plan": plan.model_dump()}, ensure_ascii=False) + "\nReturn json.",
            },
        ]
        response = await self._client.acomplete(messages, response_format={"type": "json_object"})
        if not response.content.strip():
            self._consume_reasoning_metadata("teacher", "plan_review")
            return plan
        try:
            payload = json.loads(response.content)
        except json.JSONDecodeError as exc:
            raise ValueError("Teacher plan review response was not valid JSON") from exc
        if not isinstance(payload, dict) or not payload.get("steps"):
            self._consume_reasoning_metadata("teacher", "plan_review")
            return plan
        if response.reasoning:
            self._record_reasoning("teacher", "plan_review", response.reasoning)
        normalised = self._normalise_plan_payload(payload)
        reviewed = Plan.model_validate(normalised)
        execution_mode = self._coerce_execution_mode(normalised.get("execution_mode"))
        if execution_mode is None:
            execution_mode = self._infer_execution_mode(reviewed)
        reviewed = reviewed.model_copy(update={"execution_mode": execution_mode})
        self._plan_cache[cache_key] = (now, reviewed)
        self._consume_reasoning_metadata("teacher", "plan_review")
        return reviewed

    async def avalidate_step(self, step: Step, trace: str, output: str) -> Dict[str, Any]:
        context = ExecutionContext.get()
        context.metadata["active_actor"] = "teacher"
        context.metadata["_reasoning_origin"] = ("teacher", "validation")
        structured_output = self._parse_executor_output(output)
        messages = [
            {"role": "system", "content": self._validation_prompt},
            {
                "role": "user",
                "content": json.dumps(self._build_validation_payload(step, trace, structured_output), ensure_ascii=False)
                + "\nReturn json.",
            },
        ]
        response = await self._client.acomplete(messages, response_format={"type": "json_object"})
        parsed = json.loads(response.content)
        result = {
            "valid": bool(parsed.get("valid", False)),
            "rationale": parsed.get("rationale", ""),
        }
        if response.reasoning:
            result["reasoning"] = response.reasoning
            self._record_reasoning("teacher", f"validation:{step.id}", response.reasoning)
        self._consume_reasoning_metadata("teacher", "validation")
        artifacts = structured_output.get("artifacts") if isinstance(structured_output, dict) else {}
        status = structured_output.get("status") if isinstance(structured_output, dict) else None
        if not result["valid"] and status == "ok" and isinstance(artifacts, dict) and artifacts:
            note = "Auto-validated: required artifacts detected."
            rationale = result.get("rationale") or ""
            result["rationale"] = f"{rationale} {note}".strip()
            result["valid"] = True
            result["auto_validated"] = True
        result["status"] = status
        result["artifacts"] = artifacts if isinstance(artifacts, dict) else {}
        return result

    async def agenerate_guidance(self, step: Step, evaluation: Dict[str, Any]) -> str:
        context = ExecutionContext.get()
        context.metadata["active_actor"] = "teacher"
        context.metadata["_reasoning_origin"] = ("teacher", "guidance")
        messages = [
            {"role": "system", "content": self._guidance_prompt},
            {
                "role": "user",
                "content": json.dumps(self._build_guidance_payload(step, evaluation), ensure_ascii=False),
            },
        ]
        response = await self._client.acomplete(messages)
        if response.reasoning:
            self._record_reasoning("teacher", f"guidance:{step.id}", response.reasoning)
        self._consume_reasoning_metadata("teacher", "guidance")
        return response.content

    def review_plan(self, task: str, plan: Plan) -> Plan:
        return self._run_async(self.areview_plan(task, plan))

    def validate_step(self, step: Step, trace: str, output: str) -> Dict[str, Any]:
        return self._run_async(self.avalidate_step(step, trace, output))

    def generate_guidance(self, step: Step, evaluation: Dict[str, Any]) -> str:
        return self._run_async(self.agenerate_guidance(step, evaluation))

    def collect_results(self, step_outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(step_outputs, key=lambda item: item.get("step_id", 0))

    def _build_validation_payload(
        self,
        step: Step,
        trace: str,
        structured_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        artifacts = structured_output.get("artifacts") if isinstance(structured_output, dict) else {}
        if not isinstance(artifacts, dict):
            artifacts = {}
        payload = {
            "step": step.model_dump(),
            "trace": trace,
            "status": structured_output.get("status"),
            "artifacts": artifacts,
            "notes": structured_output.get("notes"),
            "executor_output": structured_output,
        }
        return payload

    def _build_guidance_payload(self, step: Step, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "step": step.model_dump(),
            "evaluation": evaluation,
            "status": evaluation.get("status"),
            "artifacts": evaluation.get("artifacts"),
        }
        structured_output = evaluation.get("structured_output")
        if structured_output is not None:
            payload["structured_output"] = structured_output
        notes = evaluation.get("notes")
        if notes is not None:
            payload["notes"] = notes
        return payload

    def _cache_key(self, task: str, plan: Plan) -> str:
        return json.dumps({"task": task, "plan": plan.model_dump()}, sort_keys=True)

    def _run_async(self, coroutine):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)
        raise RuntimeError("Teacher synchronous methods cannot be used inside an active event loop")

    def _normalise_plan_payload(self, payload):
        if isinstance(payload, str):
            payload = json.loads(payload)
        if not isinstance(payload, dict):
            return payload
        payload.pop("total_estimated_time", None)
        mode = self._coerce_execution_mode(payload.get("execution_mode"))
        if mode is not None:
            payload["execution_mode"] = mode
        elif "execution_mode" in payload:
            payload.pop("execution_mode", None)
        steps = payload.get("steps")
        if isinstance(steps, list):
            for step in steps:
                if isinstance(step, dict):
                    step.pop("estimated_time", None)
                    step.setdefault("depends_on", [])
                    if "tool" not in step:
                        step["tool"] = None
                    if "tool_params" not in step:
                        step["tool_params"] = None
        return payload

    def _parse_executor_output(self, output: str) -> Dict[str, Any]:
        if not output:
            return {}
        try:
            parsed = json.loads(output)
        except (TypeError, json.JSONDecodeError):
            return {}
        if isinstance(parsed, dict):
            return parsed
        return {}

    def _coerce_execution_mode(self, value: Any) -> Literal["stepwise", "single_shot"] | None:
        if not isinstance(value, str):
            return None
        lowered = value.strip().lower()
        if lowered in {"stepwise", "single_shot"}:
            return cast(Literal["stepwise", "single_shot"], lowered)
        return None

    def _infer_execution_mode(self, plan: Plan) -> Literal["stepwise", "single_shot"]:
        return "single_shot" if self._plan_is_trivial(plan) else "stepwise"

    def _plan_is_trivial(self, plan: Plan) -> bool:
        if not plan.steps:
            return True
        if len(plan.steps) > 2:
            return False
        for step in plan.steps:
            if step.tool:
                return False
            if step.tool_params:
                return False
            if step.depends_on:
                return False
            if not self._is_simple_description(step.description):
                return False
        return True

    def _is_simple_description(self, description: str) -> bool:
        if not isinstance(description, str):
            return False
        text = description.strip()
        if not text:
            return False
        if len(text) > 80:
            return False
        if "\n" in text:
            return False
        for char in (";", "|", "{", "}", "[", "]"):
            if char in text:
                return False
        return True

    def _record_reasoning(self, actor: str, key: str, payload: Dict[str, Any]) -> None:
        if not payload:
            return
        context = ExecutionContext.get()
        store = context.metadata.setdefault("reasoning_traces", {})
        actor_store = store.setdefault(actor, {})
        bucket = actor_store.setdefault(key, [])
        bucket.append(payload)

    def _consume_reasoning_metadata(self, actor: str, stage: str) -> None:
        context = ExecutionContext.get()
        queue = context.metadata.get("_llm_reasoning_queue", [])
        if not queue:
            return
        remaining = [entry for entry in queue if entry.get("origin") != (actor, stage)]
        context.metadata["_llm_reasoning_queue"] = remaining
