"""Session-level reward interpretation model for the Atlas runtime."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from statistics import pstdev
from typing import Any, Dict, List, Optional, Sequence, Tuple

from atlas.config.models import RIMConfig
from atlas.evaluation.judges.base import JudgeContext
from atlas.evaluation.judges.prompts import SESSION_ARBITER_PROMPT, SESSION_REWARD_PROMPT
from atlas.runtime.orchestration.execution_context import ExecutionContext
from atlas.runtime.schema import AtlasJudgeBreakdown, AtlasJudgeSample, AtlasRewardBreakdown
from atlas.types import Step
from atlas.utils.llm_client import LLMClient

DEFAULT_TEMPERATURES: Sequence[float] = (0.2, 0.5, 0.8)


@dataclass
class SessionStepRecord:
    """Snapshot of a step within a trajectory."""

    step: Step
    trace: str
    output: str
    attempts: int = 1
    guidance: Sequence[str] | None = None
    status: str | None = None
    validation: Dict[str, Any] | None = None
    prior_results: Dict[int, Any] | None = None
    metadata: Dict[str, Any] | None = None


@dataclass
class SessionTrajectory:
    """Full execution transcript evaluated by the reward model."""

    task: str
    final_answer: str
    plan: Dict[str, Any]
    steps: Sequence[SessionStepRecord]
    execution_mode: str | None = None
    teacher_intervened: bool = False
    session_metadata: Dict[str, Any] | None = None
    focus_prompt: str | None = None


@dataclass
class RewardEvaluation:
    """Aggregated reward plus learning payloads."""

    reward: AtlasRewardBreakdown
    student_learning: str
    teacher_learning: Optional[str]


@dataclass
class SessionSample:
    """Parsed response from a single reward model invocation."""

    score: float
    uncertainty: float
    rationale: str
    principles: List[Dict[str, Any]]
    student_learning: str
    teacher_learning: Optional[str]


class Evaluator:
    """Runs tiered session-level evaluation with escalation when necessary."""

    def __init__(
        self,
        config: RIMConfig,
        *,
        small_client: LLMClient | None = None,
        large_client: LLMClient | None = None,
        focus_prompt: str | None = None,
    ) -> None:
        self._config = config
        self._temperatures = DEFAULT_TEMPERATURES
        self._variance_threshold = config.variance_threshold
        self._uncertainty_threshold = config.uncertainty_threshold
        self._small_client = small_client or LLMClient(config.small_model)
        self._arbiter_client = large_client or LLMClient(config.large_model)
        self._default_focus_prompt = focus_prompt

    async def ajudge(self, context: JudgeContext) -> AtlasRewardBreakdown:
        """Legacy entry-point used by the orchestrator during the migration period."""

        if context.reward_override is not None:
            return self._coerce_reward_breakdown(context.reward_override)
        trajectory = self._trajectory_from_context(context)
        evaluation = await self.aevaluate_session(trajectory)
        return evaluation.reward

    def judge(self, context: JudgeContext) -> AtlasRewardBreakdown:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ajudge(context))
        raise RuntimeError("Evaluator.judge cannot be invoked inside an active event loop")

    async def aevaluate_session(self, trajectory: SessionTrajectory) -> RewardEvaluation:
        focus_prompt = trajectory.focus_prompt or self._default_focus_prompt
        samples = await self._collect_samples(trajectory, focus_prompt)
        escalated = False
        escalation_reason: Optional[str] = None

        if samples and self._should_escalate(samples):
            arbiter_sample = await self._escalate_session(trajectory, samples, focus_prompt)
            if arbiter_sample is not None:
                samples = [arbiter_sample]
                escalated = True
                escalation_reason = "tier1_variance_or_uncertainty"

        if not samples:
            samples = [self._empty_sample(focus_prompt)]

        reward_breakdown, student_learning, teacher_learning = self._aggregate_samples(
            samples,
            trajectory,
            escalated=escalated,
            escalation_reason=escalation_reason,
            focus_prompt=focus_prompt,
        )
        return RewardEvaluation(reward_breakdown, student_learning, teacher_learning)

    def evaluate_session(self, trajectory: SessionTrajectory) -> RewardEvaluation:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aevaluate_session(trajectory))
        raise RuntimeError("Evaluator.evaluate_session cannot be invoked inside an active event loop")

    async def _collect_samples(
        self,
        trajectory: SessionTrajectory,
        focus_prompt: str | None,
    ) -> List[SessionSample]:
        tasks = [
            self._sample_session(trajectory, focus_prompt, temperature)
            for temperature in self._temperatures
        ]
        results = await asyncio.gather(*tasks)
        samples = [sample for sample in results if sample is not None]
        return samples

    async def _sample_session(
        self,
        trajectory: SessionTrajectory,
        focus_prompt: str | None,
        temperature: float,
    ) -> SessionSample | None:
        messages = self._build_session_messages(trajectory, focus_prompt)
        exec_context = ExecutionContext.get()
        exec_context.metadata["active_actor"] = "reward"
        exec_context.metadata["_reasoning_origin"] = ("reward", "sample")
        try:
            response = await self._small_client.acomplete(
                messages,
                response_format={"type": "json_object"},
                overrides={"temperature": temperature},
            )
        except Exception:
            exec_context.metadata.setdefault("_llm_reasoning_queue", [])
            exec_context.metadata["_llm_reasoning_queue"].clear()
            return None

        payload = self._try_parse_json(response.content)
        self._consume_reasoning_metadata("reward", "sample")

        if payload is None:
            return None
        parsed = self._parse_session_payload(payload)
        if parsed is None:
            return None

        return SessionSample(
            score=parsed["score"],
            uncertainty=parsed["uncertainty"],
            rationale=parsed["rationale"],
            principles=parsed["principles"],
            student_learning=parsed["student_learning"],
            teacher_learning=parsed["teacher_learning"],
        )

    def _should_escalate(self, samples: Sequence[SessionSample]) -> bool:
        if len(samples) < 2:
            return False
        scores = [sample.score for sample in samples]
        uncertainties = [sample.uncertainty for sample in samples]
        variance = pstdev(scores) if len(scores) > 1 else 0.0
        max_uncertainty = max(uncertainties) if uncertainties else 0.0
        return variance > self._variance_threshold or max_uncertainty > self._uncertainty_threshold

    async def _escalate_session(
        self,
        trajectory: SessionTrajectory,
        samples: Sequence[SessionSample],
        focus_prompt: str | None,
    ) -> SessionSample | None:
        tier1_summaries = []
        for index, sample in enumerate(samples, start=1):
            tier1_summaries.append(
                self._escape_for_prompt(
                    json.dumps(
                        {
                            "sample": index,
                            "score": sample.score,
                            "uncertainty": sample.uncertainty,
                            "rationale": sample.rationale,
                            "principles": sample.principles,
                            "student_learning": sample.student_learning,
                            "teacher_learning": sample.teacher_learning,
                        },
                        ensure_ascii=False,
                    )
                )
            )
        meta_prompt = SESSION_ARBITER_PROMPT.format(
            task=self._escape_for_prompt(trajectory.task),
            execution_mode=self._escape_for_prompt(str(trajectory.execution_mode)),
            teacher_intervened=self._escape_for_prompt(str(trajectory.teacher_intervened)),
            final_answer=self._escape_for_prompt(trajectory.final_answer),
            focus_prompt=self._escape_for_prompt(focus_prompt or ""),
            tier1_summaries="\n".join(tier1_summaries),
        )
        exec_context = ExecutionContext.get()
        exec_context.metadata["active_actor"] = "reward"
        exec_context.metadata["_reasoning_origin"] = ("reward", "escalation")
        response = None
        try:
            response = await self._arbiter_client.acomplete(
                messages=[{"role": "user", "content": meta_prompt}],
                response_format={"type": "json_object"},
                overrides={"temperature": 0.3},
            )
            payload = self._try_parse_json(response.content)
        except Exception:
            payload = None

        reasoning_queue = self._consume_reasoning_metadata("reward", "escalation")
        reasoning = response.reasoning if payload is not None and response else None
        if reasoning_queue:
            if reasoning:
                reasoning = {"response": reasoning, "queue": reasoning_queue}
            else:
                reasoning = {"queue": reasoning_queue}

        if payload is None:
            return None
        parsed = self._parse_session_payload(payload)
        if parsed is None:
            return None

        return SessionSample(
            score=parsed["score"],
            uncertainty=parsed["uncertainty"],
            rationale=parsed["rationale"],
            principles=parsed["principles"],
            student_learning=parsed["student_learning"],
            teacher_learning=parsed["teacher_learning"],
        )

    def _aggregate_samples(
        self,
        samples: Sequence[SessionSample],
        trajectory: SessionTrajectory,
        *,
        escalated: bool,
        escalation_reason: Optional[str],
        focus_prompt: str | None,
    ) -> Tuple[AtlasRewardBreakdown, Dict[str, Any], Optional[Dict[str, Any]]]:
        scores = [sample.score for sample in samples]
        aggregated_score = sum(scores) / len(scores)
        best_sample = min(samples, key=lambda sample: sample.uncertainty)
        judge_samples = [
            AtlasJudgeSample(
                score=sample.score,
                rationale=sample.rationale,
                principles=sample.principles,
                uncertainty=sample.uncertainty,
                temperature=None,
            )
            for sample in samples
        ]
        judge_breakdown = AtlasJudgeBreakdown(
            identifier="session_reward",
            score=aggregated_score,
            rationale=best_sample.rationale,
            principles=best_sample.principles,
            samples=judge_samples,
            escalated=escalated,
            escalation_reason=escalation_reason,
        )
        raw_samples = [
            {
                "score": sample.score,
                "uncertainty": sample.uncertainty,
            }
            for sample in samples
        ]
        reward_raw = {
            "score": aggregated_score,
            "samples": raw_samples,
            "execution_mode": trajectory.execution_mode,
            "teacher_intervened": trajectory.teacher_intervened,
            "focus_prompt": focus_prompt,
        }
        reward = AtlasRewardBreakdown(
            score=aggregated_score,
            judges=[judge_breakdown],
            rationale=None,
            raw=reward_raw,
        )

        student_learning = best_sample.student_learning

        teacher_learning = None
        if (trajectory.execution_mode or "").lower() != "auto":
            teacher_text = (best_sample.teacher_learning or "").strip()
            if teacher_text:
                teacher_learning = teacher_text
            else:
                teacher_learning = None

        return reward, student_learning, teacher_learning

    def _build_session_messages(
        self,
        trajectory: SessionTrajectory,
        focus_prompt: str | None,
    ) -> List[Dict[str, str]]:
        system_prompt = SESSION_REWARD_PROMPT.format(
            task=self._escape_for_prompt(trajectory.task),
            execution_mode=self._escape_for_prompt(str(trajectory.execution_mode)),
            teacher_intervened=self._escape_for_prompt(str(trajectory.teacher_intervened)),
            focus_prompt=self._escape_for_prompt(focus_prompt or ""),
            plan=self._escape_for_prompt(json.dumps(trajectory.plan or {}, ensure_ascii=False)),
            final_answer=self._escape_for_prompt(trajectory.final_answer),
            session_metadata=self._escape_for_prompt(json.dumps(trajectory.session_metadata or {}, ensure_ascii=False)),
        )
        user_payload = json.dumps(
            {
                "task": trajectory.task,
                "execution_mode": trajectory.execution_mode,
                "teacher_intervened": trajectory.teacher_intervened,
                "plan": trajectory.plan,
                "final_answer": trajectory.final_answer,
                "session_metadata": trajectory.session_metadata or {},
                "focus_prompt": focus_prompt,
            },
            ensure_ascii=False,
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ]

    def _trajectory_from_context(self, context: JudgeContext) -> SessionTrajectory:
        record = SessionStepRecord(
            step=context.step,
            trace=context.trace,
            output=context.output,
            attempts=context.attempt,
            guidance=list(context.guidance or []),
            status=context.status,
            validation=context.validation,
            prior_results=context.prior_results,
        )
        plan_payload: Dict[str, Any] = {"steps": [{"id": context.step.id, "description": context.step.description}]}
        return SessionTrajectory(
            task=context.task,
            final_answer=context.output,
            plan=plan_payload,
            steps=[record],
            execution_mode=context.execution_mode,
            teacher_intervened=context.teacher_intervened or bool(context.guidance),
            session_metadata=None,
            focus_prompt=context.focus_prompt or self._default_focus_prompt,
        )

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
    def _parse_session_payload(payload: Dict[str, Any]) -> Dict[str, Any] | None:
        score = payload.get("score")
        uncertainty = payload.get("uncertainty")
        rationale = payload.get("rationale", "")
        principles_payload = payload.get("principles") or []
        if not isinstance(score, (int, float)) or not isinstance(uncertainty, (int, float)):
            return None
        principles: List[Dict[str, Any]] = []
        if isinstance(principles_payload, list):
            for entry in principles_payload:
                if isinstance(entry, dict):
                    name = entry.get("name")
                    weight = entry.get("weight")
                    description = entry.get("description", "")
                    if isinstance(name, str) and isinstance(weight, (int, float)):
                        principles.append(
                            {
                                "name": name,
                                "weight": float(weight),
                                "description": description if isinstance(description, str) else str(description),
                            }
                        )
        student_learning = payload.get("student_learning")
        if not isinstance(student_learning, str):
            student_learning = ""
        teacher_learning_value = payload.get("teacher_learning")
        if isinstance(teacher_learning_value, str):
            teacher_learning: Optional[str] = teacher_learning_value
        elif teacher_learning_value is None:
            teacher_learning = None
        else:
            teacher_learning = None
        return {
            "score": float(score),
            "uncertainty": float(uncertainty),
            "rationale": rationale if isinstance(rationale, str) else str(rationale),
            "principles": principles,
            "student_learning": student_learning,
            "teacher_learning": teacher_learning,
        }

    @staticmethod
    def _empty_sample(focus_prompt: str | None) -> SessionSample:
        return SessionSample(
            score=0.0,
            uncertainty=1.0,
            rationale="No valid reward sample produced.",
            principles=[],
            student_learning="",
            teacher_learning=None,
        )

    def _coerce_reward_breakdown(
        self,
        payload: AtlasRewardBreakdown | Dict[str, Any],
    ) -> AtlasRewardBreakdown:
        if isinstance(payload, AtlasRewardBreakdown):
            return payload
        return AtlasRewardBreakdown.from_dict(payload)

    @staticmethod
    def _escape_for_prompt(value: Any) -> str:
        text = "" if value is None else str(value)
        return text.replace("{", "{{").replace("}", "}}")

    @staticmethod
    def _consume_reasoning_metadata(actor: str, stage: str) -> List[Dict[str, Any]]:
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
