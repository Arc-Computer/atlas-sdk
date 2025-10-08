"""Sequential reward interpretation model for Atlas SDK."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from statistics import pstdev
from typing import Any, Dict, List, Sequence

from atlas.config.models import RIMConfig
from atlas.runtime.schema import AtlasJudgeBreakdown, AtlasJudgeSample, AtlasRewardBreakdown
from atlas.reward.helpfulness_judge import HelpfulnessJudge
from atlas.reward.judge import Judge, JudgeContext, JudgeOutcome, JudgeSample
from atlas.reward.process_judge import ProcessJudge
from atlas.utils.llm_client import LLMClient
from atlas.orchestration.execution_context import ExecutionContext

_DEFAULT_TEMPERATURES: Sequence[float] = (0.2, 0.5, 0.8)


@dataclass
class _JudgeState:
    judge: Judge
    outcome: JudgeOutcome


class Evaluator:
    """Runs tiered RIM evaluation with escalation when necessary."""

    def __init__(
        self,
        config: RIMConfig,
        *,
        small_client: LLMClient | None = None,
        large_client: LLMClient | None = None,
    ) -> None:
        self._config = config
        self._temperatures = _DEFAULT_TEMPERATURES
        self._variance_threshold = config.variance_threshold
        self._uncertainty_threshold = config.uncertainty_threshold
        self._small_client = small_client or LLMClient(config.small_model)
        self._arbiter_client = large_client or LLMClient(config.large_model)
        self._available_judges: Dict[str, Judge] = {
            "process": ProcessJudge(self._small_client),
            "helpfulness": HelpfulnessJudge(self._small_client),
        }
        active = [name for name, enabled in (config.active_judges or {}).items() if enabled]
        if not active:
            active = list(self._available_judges.keys())
        self._judges = [self._available_judges[name] for name in active if name in self._available_judges]
        if not self._judges:
            raise ValueError("No active RIM judges are available")

    async def ajudge(self, context: JudgeContext) -> AtlasRewardBreakdown:
        judge_states = await asyncio.gather(
            *(self._evaluate_judge(judge, context) for judge in self._judges)
        )
        aggregated = self._aggregate(judge_states)

        raw_judges: List[Dict[str, Any]] = []
        judges: List[AtlasJudgeBreakdown] = []
        for state in judge_states:
            samples = [
                AtlasJudgeSample(
                    score=sample.score,
                    rationale=sample.rationale,
                    principles=sample.principles,
                    uncertainty=sample.uncertainty,
                    temperature=sample.temperature,
                )
                for sample in state.outcome.samples
            ]
            judges.append(
                AtlasJudgeBreakdown(
                    identifier=state.judge.identifier,
                    score=state.outcome.score,
                    rationale=state.outcome.rationale,
                    principles=state.outcome.principles,
                    samples=samples,
                    escalated=state.outcome.escalated,
                    escalation_reason=state.outcome.escalation_reason,
                )
            )
            raw_judges.append(
                {
                    "identifier": state.judge.identifier,
                    "score": state.outcome.score,
                    "rationale": state.outcome.rationale,
                    "principles": state.outcome.principles,
                    "samples": [
                        {
                            "score": sample.score,
                            "rationale": sample.rationale,
                            "principles": sample.principles,
                            "uncertainty": sample.uncertainty,
                            "temperature": sample.temperature,
                            "reasoning": sample.reasoning,
                        }
                        for sample in state.outcome.samples
                    ],
                    "escalated": state.outcome.escalated,
                    "escalation_reason": state.outcome.escalation_reason,
                    "reasoning": state.outcome.reasoning,
                }
            )

        raw_payload = {"score": aggregated, "judges": raw_judges}
        return AtlasRewardBreakdown(
            score=aggregated,
            judges=judges,
            rationale=None,
            raw=raw_payload,
        )

    def judge(self, context: JudgeContext) -> AtlasRewardBreakdown:
        return asyncio.run(self.ajudge(context))

    async def _evaluate_judge(self, judge: Judge, context: JudgeContext) -> _JudgeState:
        samples = await self._collect_samples(judge, context)
        if not samples:
            outcome = await self._escalate(judge, context, samples, "tier1_no_valid_samples")
            return _JudgeState(judge=judge, outcome=outcome)

        scores = [sample.score for sample in samples]
        uncertainties = [sample.uncertainty for sample in samples]
        variance = pstdev(scores) if len(scores) > 1 else 0.0
        max_uncertainty = max(uncertainties) if uncertainties else 0.0

        escalate = (
            variance > self._variance_threshold
            or max_uncertainty > self._uncertainty_threshold
        )

        if escalate:
            outcome = await self._escalate(judge, context, samples, "tier1_variance_or_uncertainty")
        else:
            best_sample = min(samples, key=lambda s: s.uncertainty)
            outcome = JudgeOutcome(
                identifier=judge.identifier,
                score=max(0.0, min(best_sample.score, 1.0)),
                rationale=best_sample.rationale,
                principles=best_sample.principles,
                samples=samples,
                escalated=False,
                escalation_reason=None,
                reasoning=best_sample.reasoning,
            )
        for sample in samples:
            if sample.reasoning:
                self._record_reasoning(f"{judge.identifier}:sample:{sample.temperature}", sample.reasoning)
        if outcome.reasoning:
            self._record_reasoning(f"{judge.identifier}:outcome", outcome.reasoning)
        return _JudgeState(judge=judge, outcome=outcome)

    async def _collect_samples(self, judge: Judge, context: JudgeContext) -> List[JudgeSample]:
        tasks = [judge.asample(context, temp) for temp in self._temperatures]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        samples: List[JudgeSample] = []
        for result in results:
            if isinstance(result, JudgeSample):
                samples.append(result)
        return samples

    async def _escalate(
        self,
        judge: Judge,
        context: JudgeContext,
        samples: List[JudgeSample],
        reason: str,
    ) -> JudgeOutcome:
        meta_prompt = judge.build_meta_prompt(context, samples, reason)
        response = None
        exec_context = ExecutionContext.get()
        exec_context.metadata["active_actor"] = "reward"
        exec_context.metadata["_reasoning_origin"] = ("reward", "escalation")
        try:
            response = await self._arbiter_client.acomplete(
                messages=[{"role": "user", "content": meta_prompt}],
                response_format={"type": "json_object"},
                overrides={"temperature": 0.3},
            )
            payload = json.loads(response.content)
        except Exception:
            payload = None

        if isinstance(payload, dict):
            score = payload.get("score")
            rationale = payload.get("rationale", payload.get("explanation", ""))
            uncertainty = payload.get("uncertainty")
            principles = payload.get("principles", [])
            parsed_principles = judge._normalize_principles(principles)
            if not parsed_principles and samples:
                parsed_principles = samples[0].principles
            rationale_text = rationale if isinstance(rationale, str) else ""
            final_score = score if isinstance(score, (int, float)) else 0.0
        else:
            final_score = 0.0
            rationale_text = "Arbiter model failed to produce a valid response."
            parsed_principles = samples[0].principles if samples else []

        outcome = JudgeOutcome(
            identifier=judge.identifier,
            score=max(0.0, min(float(final_score), 1.0)),
            rationale=rationale_text,
            principles=parsed_principles,
            samples=samples,
            escalated=True,
            escalation_reason=reason,
            reasoning=self._merge_reasoning_payload(
                response.reasoning if response else None,
                self._consume_reasoning_metadata("reward", "escalation"),
            ),
        )
        return outcome

    def _aggregate(self, states: Sequence[_JudgeState]) -> float:
        if not states:
            return 0.0
        return sum(state.outcome.score for state in states) / len(states)

    def _merge_reasoning_payload(
        self,
        base: Dict[str, Any] | None,
        queue_entries: List[Dict[str, Any]],
    ) -> Dict[str, Any] | None:
        if queue_entries:
            if base:
                return {"response": base, "queue": queue_entries}
            return {"queue": queue_entries}
        return base

    def _record_reasoning(self, key: str, payload: Dict[str, Any] | None) -> None:
        if not payload:
            return
        context = ExecutionContext.get()
        store = context.metadata.setdefault("reasoning_traces", {})
        actor_store = store.setdefault("reward", {})
        bucket = actor_store.setdefault(key, [])
        bucket.append(payload)

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
