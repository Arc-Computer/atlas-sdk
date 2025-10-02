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
        messages = [
            {"role": "system", "content": self._plan_prompt},
            {"role": "user", "content": json.dumps({"task": task, "plan": plan.model_dump()}, ensure_ascii=False)},
        ]
        response = await self._client.acomplete(messages, response_format={"type": "json_object"})
        reviewed = Plan.model_validate_json(response.content)
        self._plan_cache[cache_key] = (now, reviewed)
        return reviewed

    async def avalidate_step(self, step: Step, trace: str, output: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": self._validation_prompt},
            {"role": "user", "content": json.dumps(self._build_validation_payload(step, trace, output), ensure_ascii=False)},
        ]
        response = await self._client.acomplete(messages, response_format={"type": "json_object"})
        parsed = json.loads(response.content)
        return {
            "valid": bool(parsed.get("valid", False)),
            "rationale": parsed.get("rationale", ""),
        }

    async def agenerate_guidance(self, step: Step, evaluation: Dict[str, Any]) -> str:
        messages = [
            {"role": "system", "content": self._guidance_prompt},
            {"role": "user", "content": json.dumps(self._build_guidance_payload(step, evaluation), ensure_ascii=False)},
        ]
        response = await self._client.acomplete(messages)
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
