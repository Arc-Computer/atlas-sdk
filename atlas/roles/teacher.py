"""Teacher responsible for plan review, validation, and guidance."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

from atlas.config.models import TeacherConfig
from atlas.transition.rewriter import RewrittenTeacherPrompts
from atlas.types import Plan
from atlas.types import Step
from atlas.utils.llm_client import LLMClient
from atlas.orchestration.execution_context import ExecutionContext


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
        try:
            payload = json.loads(response.content)
        except json.JSONDecodeError as exc:
            raise ValueError("Teacher plan review response was not valid JSON") from exc
        if not isinstance(payload, dict) or not payload.get("steps"):
            self._consume_reasoning_metadata("teacher", "plan_review")
            return plan
        if response.reasoning:
            self._record_reasoning("teacher", "plan_review", response.reasoning)
        reviewed = Plan.model_validate(self._normalise_plan_payload(payload))
        self._plan_cache[cache_key] = (now, reviewed)
        self._consume_reasoning_metadata("teacher", "plan_review")
        return reviewed

    async def avalidate_step(self, step: Step, trace: str, output: str) -> Dict[str, Any]:
        context = ExecutionContext.get()
        context.metadata["active_actor"] = "teacher"
        context.metadata["_reasoning_origin"] = ("teacher", "validation")
        messages = [
            {"role": "system", "content": self._validation_prompt},
            {
                "role": "user",
                "content": json.dumps(self._build_validation_payload(step, trace, output), ensure_ascii=False)
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

    def _build_validation_payload(self, step: Step, trace: str, output: str) -> Dict[str, Any]:
        return {
            "step": step.model_dump(),
            "trace": trace,
            "output": output,
        }

    def _build_guidance_payload(self, step: Step, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "step": step.model_dump(),
            "evaluation": evaluation,
        }

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
