"""Teacher responsible for plan review, validation, and guidance."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Sequence
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

    async def avalidate_step(
        self,
        step: Step,
        trace: str,
        structured_output: Dict[str, Any],
        prior_results: Dict[int, Any],
        prior_guidance: Sequence[str],
        attempt_guidance: Sequence[str],
    ) -> Dict[str, Any]:
        context = ExecutionContext.get()
        context.metadata["active_actor"] = "teacher"
        context.metadata["_reasoning_origin"] = ("teacher", "validation")
        messages = [
            {"role": "system", "content": self._validation_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    self._build_validation_payload(
                        step,
                        trace,
                        structured_output,
                        prior_results,
                        prior_guidance,
                        attempt_guidance,
                    ),
                    ensure_ascii=False,
                )
                + "\nReturn json.",
            },
        ]
        response = await self._client.acomplete(messages, response_format={"type": "json_object"})
        parsed = json.loads(response.content)
        result: Dict[str, Any] = {
            "valid": bool(parsed.get("valid", False)),
            "guidance": parsed.get("guidance"),
        }
        if response.reasoning:
            self._record_reasoning("teacher", f"validation:{step.id}", response.reasoning)
            result["reasoning"] = response.reasoning
        self._consume_reasoning_metadata("teacher", "validation")
        result["status"] = structured_output.get("status")
        artifacts = structured_output.get("artifacts")
        if artifacts is not None:
            result["artifacts"] = artifacts
        deliverable = structured_output.get("deliverable")
        if deliverable is not None:
            result["deliverable"] = deliverable
        text_output = structured_output.get("text")
        if text_output is not None:
            result["text"] = text_output
        reason = structured_output.get("reason")
        if reason is not None:
            result["reason"] = reason
        return result

    async def agenerate_guidance(self, step: Step, evaluation: Dict[str, Any]) -> str:
        context = ExecutionContext.get()
        context.metadata["active_actor"] = "teacher"
        context.metadata["_reasoning_origin"] = ("teacher", "guidance")
        guidance = None
        if isinstance(evaluation, dict):
            validation_payload = evaluation.get("validation") if "validation" in evaluation else evaluation
            if isinstance(validation_payload, dict):
                candidate = validation_payload.get("guidance")
                if isinstance(candidate, str):
                    guidance = candidate
        self._consume_reasoning_metadata("teacher", "guidance")
        if guidance is None:
            guidance = ""
        return guidance

    def review_plan(self, task: str, plan: Plan) -> Plan:
        return self._run_async(self.areview_plan(task, plan))

    def validate_step(
        self,
        step: Step,
        trace: str,
        structured_output: Dict[str, Any],
        prior_results: Dict[int, Any],
        prior_guidance: Sequence[str],
        attempt_guidance: Sequence[str],
    ) -> Dict[str, Any]:
        return self._run_async(
            self.avalidate_step(step, trace, structured_output, prior_results, prior_guidance, attempt_guidance)
        )

    def generate_guidance(self, step: Step, evaluation: Dict[str, Any]) -> str:
        return self._run_async(self.agenerate_guidance(step, evaluation))

    def collect_results(self, step_outputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return sorted(step_outputs, key=lambda item: item.get("step_id", 0))

    def _build_validation_payload(
        self,
        step: Step,
        trace: str,
        structured_output: Dict[str, Any],
        prior_results: Dict[int, Any],
        prior_guidance: Sequence[str],
        attempt_guidance: Sequence[str],
    ) -> Dict[str, Any]:
        payload = {
            "step": step.model_dump(),
            "trace": trace,
            "status": structured_output.get("status"),
            "student_response": structured_output.get("text"),
            "deliverable": self._jsonify(structured_output.get("deliverable")),
            "artifacts": self._jsonify(structured_output.get("artifacts")),
            "result": self._jsonify(structured_output.get("result")),
            "reason": structured_output.get("reason"),
            "structured_output": self._jsonify(structured_output),
            "validated_context": self._jsonify(prior_results),
            "prior_guidance": list(prior_guidance),
            "attempt_guidance": list(attempt_guidance),
        }
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

    def _jsonify(self, value: Any, depth: int = 0) -> Any:
        if depth > 6:
            return str(value)
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): self._jsonify(item, depth + 1) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._jsonify(item, depth + 1) for item in value]
        if hasattr(value, "model_dump"):
            try:
                dumped = value.model_dump()
            except Exception:
                return str(value)
            return self._jsonify(dumped, depth + 1)
        if hasattr(value, "__dict__"):
            return self._jsonify(vars(value), depth + 1)
        return str(value)
