"""Judge primitives powering the Atlas RIM evaluator."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from atlas.types import Step
from atlas.utils.llm_client import LLMClient
from atlas.orchestration.execution_context import ExecutionContext


@dataclass
class JudgeContext:
    """Inputs describing the step being evaluated."""

    task: str
    step: Step
    trace: str
    output: str
    attempt: int = 1
    prior_results: Dict[int, str] | None = None
    guidance: Sequence[str] | None = None


@dataclass
class JudgeSample:
    """One sampling pass from a judge's small model."""

    score: float
    rationale: str
    principles: List[Dict[str, Any]]
    uncertainty: float
    temperature: float
    reasoning: Dict[str, Any] | None = None


@dataclass
class JudgeOutcome:
    """Finalised outcome after aggregation/escalation."""

    identifier: str
    score: float
    rationale: str
    principles: List[Dict[str, Any]]
    samples: List[JudgeSample]
    escalated: bool
    escalation_reason: str | None
    reasoning: Dict[str, Any] | None = None


class Judge:
    """Base class for RIM judges."""

    def __init__(self, identifier: str, client: LLMClient) -> None:
        self.identifier = identifier
        self.weight = 1.0
        self._client = client

    async def asample(self, context: JudgeContext, temperature: float) -> JudgeSample | None:
        """Generate a single sample using the small judge model."""

        messages = self._build_messages(context)
        overrides = {"temperature": temperature}
        execution_context = ExecutionContext.get()
        execution_context.metadata["active_actor"] = "reward"
        execution_context.metadata["_reasoning_origin"] = ("reward", "sample")
        try:
            response = await self._client.acomplete(
                messages,
                response_format={"type": "json_object"},
                overrides=overrides,
            )
        except Exception:
            self._consume_reasoning_metadata("reward", "sample")
            return None

        payload = self._try_parse_json(response.content)
        if payload is None:
            self._consume_reasoning_metadata("reward", "sample")
            return None

        score = payload.get("score")
        uncertainty = payload.get("uncertainty")
        rationale = payload.get("rationale", payload.get("explanation", ""))
        principles = payload.get("principles", [])

        if not isinstance(score, (int, float)) or not isinstance(uncertainty, (int, float)):
            return None

        parsed_principles = self._normalize_principles(principles)

        rationale_text = rationale if isinstance(rationale, str) else ""

        queue_entries = self._consume_reasoning_metadata("reward", "sample")
        combined_reasoning = response.reasoning or payload.get("reasoning")
        if queue_entries:
            if combined_reasoning:
                combined_reasoning = {
                    "response": combined_reasoning,
                    "queue": queue_entries,
                }
            else:
                combined_reasoning = {"queue": queue_entries}

        sample = JudgeSample(
            score=float(score),
            rationale=rationale_text,
            principles=parsed_principles,
            uncertainty=float(uncertainty),
            temperature=temperature,
            reasoning=combined_reasoning or None,
        )
        return sample

    async def ajudge(self, context: JudgeContext) -> JudgeOutcome:
        """Must be implemented by concrete judges."""

        raise NotImplementedError

    def build_meta_prompt(
        self,
        context: JudgeContext,
        samples: List[JudgeSample],
        escalation_reason: str | None,
    ) -> str:
        """Return the meta prompt shown to the arbiter model."""

        raise NotImplementedError

    def judge(self, context: JudgeContext) -> JudgeOutcome:
        return asyncio.run(self.ajudge(context))

    def _build_messages(self, context: JudgeContext) -> Sequence[Dict[str, Any]]:
        raise NotImplementedError

    @staticmethod
    def _try_parse_json(payload: Any) -> Dict[str, Any] | None:
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, str):
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                return None
        return None

    @staticmethod
    def _normalize_principles(value: Any) -> List[Dict[str, Any]]:
        if not isinstance(value, list):
            return []
        normalised: List[Dict[str, Any]] = []
        for entry in value:
            if isinstance(entry, dict):
                name = entry.get("name")
                weight = entry.get("weight")
                description = entry.get("description", "")
                if isinstance(name, str) and isinstance(weight, (int, float)):
                    normalised.append(
                        {
                            "name": name,
                            "weight": float(weight),
                            "description": description if isinstance(description, str) else "",
                        }
                    )
            elif isinstance(entry, str):
                normalised.append({"name": entry, "weight": 1.0, "description": entry})
        if not normalised:
            return []
        weight_sum = sum(item["weight"] for item in normalised)
        if weight_sum > 0:
            for item in normalised:
                item["weight"] = item["weight"] / weight_sum
        return normalised

    def _consume_reasoning_metadata(self, actor: str, stage: str) -> List[Dict[str, Any]]:
        context = ExecutionContext.get()
        queue = context.metadata.get("_llm_reasoning_queue", [])
        if not queue:
            return []
        matched: List[Dict[str, Any]] = []
        remaining: List[Dict[str, Any]] = []
        for entry in queue:
            if entry.get("origin") == (actor, stage):
                matched.append(entry.get("payload") or {})
            else:
                remaining.append(entry)
        context.metadata["_llm_reasoning_queue"] = remaining
        return matched
