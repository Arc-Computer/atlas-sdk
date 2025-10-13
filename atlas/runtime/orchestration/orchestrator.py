"""Sequential orchestrator coordinating Teacher, Student, and RIM evaluation."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from uuid import uuid4

from atlas.config.models import AdaptiveTeachingConfig
from atlas.config.models import OrchestrationConfig
from atlas.config.models import RIMConfig
from atlas.runtime.models import IntermediateStepPayload
from atlas.runtime.models import IntermediateStepType
from atlas.runtime.models import StreamEventData
from atlas.runtime.orchestration.dependency_graph import DependencyGraph
from atlas.runtime.orchestration.execution_context import ExecutionContext
from atlas.evaluation.evaluator import Evaluator
from atlas.evaluation.judges.base import JudgeContext
from atlas.personas.student import Student
from atlas.personas.student import StudentStepResult
from atlas.personas.teacher import Teacher
from atlas.runtime.schema import AtlasRewardBreakdown
from atlas.utils.triage import TriageDossier, default_build_dossier
from atlas.types import Plan
from atlas.types import Result
from atlas.types import Step
from atlas.types import StepEvaluation
from atlas.types import StepResult

logger = logging.getLogger(__name__)


@dataclass
class _StepExecutionOutcome:
    result: StudentStepResult
    evaluation: StepEvaluation
    attempts: int
    context_entry: Dict[str, Any] | None
    reward_skipped: bool
    status: str
    artifacts: Dict[str, Any]
    deliverable: Any | None = None


@dataclass(slots=True)
class CapabilityProbeResult:
    mode: Optional[str]
    confidence: Optional[float]
    evidence: List[str] = field(default_factory=list)
    recommended_personas: List[str] = field(default_factory=list)
    raw: Dict[str, Any] | None = None
    status: str | None = None


@dataclass(slots=True)
class AdaptiveModeDecision:
    mode: str
    confidence: Optional[float] = None
    reason: str | None = None
    evidence: List[str] = field(default_factory=list)
    certification: bool = False
    probe: CapabilityProbeResult | None = None


class Orchestrator:
    def __init__(
        self,
        teacher: Teacher,
        student: Student,
        evaluator: Evaluator,
        orchestration_config: OrchestrationConfig,
        rim_config: RIMConfig,
        adaptive_config: AdaptiveTeachingConfig | None = None,
        triage_adapter: Callable[[str, Dict[str, Any] | None], TriageDossier] | None = None,
        persona_refresh: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._teacher = teacher
        self._student = student
        self._evaluator = evaluator
        self._orchestration = orchestration_config
        self._rim_config = rim_config
        self._rim_retry_threshold = getattr(rim_config, "retry_threshold", 0.6)
        self._persona_refresh = persona_refresh
        self._adaptive = adaptive_config or AdaptiveTeachingConfig()
        self._triage_adapter = triage_adapter or default_build_dossier
        self._seen_fingerprints: Set[str] = set()
        self._current_retry_limit: int = self._orchestration.max_retries

    async def arun(self, task: str) -> Result:
        context = ExecutionContext.get()
        context.mark_certification_run(False)
        self._current_retry_limit = self._orchestration.max_retries
        manager = context.intermediate_step_manager
        orchestration_id = str(uuid4())
        manager.push_intermediate_step(
            IntermediateStepPayload(
                UUID=orchestration_id,
                event_type=IntermediateStepType.WORKFLOW_START,
                name="orchestration",
                data=StreamEventData(input={"task": task}),
            )
        )
        dossier = self._build_triage_dossier(task, context)
        context.set_triage_dossier(dossier)
        initial_plan = await self._student.acreate_plan(task)
        reviewed_plan = await self._teacher.areview_plan(task, initial_plan)
        plan_payload = reviewed_plan.model_dump()
        execution_mode = getattr(reviewed_plan, "execution_mode", "stepwise")
        context.metadata["task"] = task
        context.metadata["plan"] = plan_payload
        context.metadata["original_plan"] = plan_payload
        context.metadata["execution_mode"] = execution_mode
        context.metadata.pop("single_shot", None)
        if self._persona_refresh is not None:
            await self._persona_refresh()
        final_plan, decision = await self._apply_adaptive_mode(task, context, reviewed_plan, dossier)
        context.metadata["plan"] = final_plan.model_dump()
        context.metadata["execution_mode"] = final_plan.execution_mode

        if final_plan.execution_mode == "single_shot":
            context.metadata["single_shot"] = True
            single_step = final_plan.steps[0]
            context_outputs: Dict[int, Dict[str, Any]] = {}
            step_summaries: List[Dict[str, Any]] = []
            step_results: List[StepResult] = []

            outcome = await self._run_step(task, single_step, context_outputs, context)
            if outcome.context_entry is not None:
                context_outputs[single_step.id] = outcome.context_entry
            result = outcome.result
            evaluation = outcome.evaluation
            attempts = outcome.attempts
            step_summaries.append(
                {
                    "step_id": single_step.id,
                    "description": single_step.description,
                    "status": outcome.status,
                    "output": result.output,
                    "artifacts": outcome.artifacts,
                    "deliverable": outcome.deliverable,
                    "reason": result.metadata.get("reason"),
                }
            )
            step_results.append(
                StepResult(
                    step_id=single_step.id,
                    trace=result.trace,
                    output=result.output,
                    evaluation=evaluation,
                    attempts=attempts,
                    metadata=result.metadata,
                )
            )
            organized_results = self._teacher.collect_results(step_summaries)
            context.metadata["single_shot_results"] = organized_results
            final_answer = await self._student.asynthesize_final_answer(task, organized_results)
            manager.push_intermediate_step(
                IntermediateStepPayload(
                    UUID=orchestration_id,
                    event_type=IntermediateStepType.WORKFLOW_END,
                    name="orchestration",
                    data=StreamEventData(output=final_answer),
                )
            )
            return Result(final_answer=final_answer, plan=final_plan, step_results=step_results)

        context.metadata.pop("single_shot", None)
        levels = self._determine_levels(final_plan)
        context_outputs: Dict[int, Dict[str, Any]] = {}
        step_summaries: List[Dict[str, Any]] = []
        step_results: List[StepResult] = []
        for level in levels:
            if len(level) == 1:
                step_id = level[0]
                step = self._lookup_step(final_plan, step_id)
                outcome = await self._run_step(task, step, context_outputs, context)
                if outcome.context_entry is not None:
                    context_outputs[step.id] = outcome.context_entry
                result = outcome.result
                evaluation = outcome.evaluation
                attempts = outcome.attempts
                step_summaries.append(
                    {
                        "step_id": step.id,
                        "description": step.description,
                        "status": outcome.status,
                        "output": result.output,
                        "artifacts": outcome.artifacts,
                        "deliverable": result.deliverable,
                        "reason": result.metadata.get("reason"),
                        "trace": result.trace,
                        "evaluation": evaluation.to_dict(),
                        "metadata": result.metadata,
                        "attempts": attempts,
                    }
                )
                step_results.append(
                    StepResult(
                        step_id=step.id,
                        trace=result.trace,
                        output=result.output,
                        evaluation=evaluation,
                        attempts=attempts,
                        metadata=result.metadata,
                    )
                )
            else:
                steps = [self._lookup_step(final_plan, step_id) for step_id in level]
                tasks = [
                    self._run_step(task, step, dict(context_outputs), context)
                    for step in steps
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                captured_exception: Exception | None = None
                for step, outcome in zip(steps, results):
                    if isinstance(outcome, Exception):
                        evaluation = self._build_error_evaluation(str(outcome))
                        step_summaries.append(
                            {
                                "step_id": step.id,
                                "description": step.description,
                                "output": "",
                                "trace": "",
                                "evaluation": evaluation.to_dict(),
                                "metadata": {},
                                "attempts": 0,
                            }
                        )
                        step_results.append(
                            StepResult(
                                step_id=step.id,
                                trace="",
                                output="",
                                evaluation=evaluation,
                                attempts=0,
                                metadata={},
                            )
                        )
                        if captured_exception is None:
                            captured_exception = outcome
                        continue

                    if outcome.context_entry is not None:
                        context_outputs[step.id] = outcome.context_entry
                    result = outcome.result
                    evaluation = outcome.evaluation
                    attempts = outcome.attempts
                    step_summaries.append(
                        {
                            "step_id": step.id,
                            "description": step.description,
                            "status": outcome.status,
                            "output": result.output,
                            "artifacts": outcome.artifacts,
                            "deliverable": result.deliverable,
                            "reason": result.metadata.get("reason"),
                            "trace": result.trace,
                            "evaluation": evaluation.to_dict(),
                            "metadata": result.metadata,
                            "attempts": attempts,
                        }
                    )
                    step_results.append(
                        StepResult(
                            step_id=step.id,
                            trace=result.trace,
                            output=result.output,
                            evaluation=evaluation,
                            attempts=attempts,
                            metadata=result.metadata,
                        )
                    )
                if captured_exception is not None:
                    raise captured_exception
        organized_results = self._teacher.collect_results(step_summaries)
        final_answer = await self._student.asynthesize_final_answer(task, organized_results)
        manager.push_intermediate_step(
            IntermediateStepPayload(
                UUID=orchestration_id,
                event_type=IntermediateStepType.WORKFLOW_END,
                name="orchestration",
                data=StreamEventData(output=final_answer),
            )
        )
        return Result(final_answer=final_answer, plan=final_plan, step_results=step_results)

    def run(self, task: str) -> Result:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.arun(task))
        raise RuntimeError("Orchestrator synchronous entry cannot run inside an active event loop")

    async def _run_step(
        self,
        task: str,
        step: Step,
        context_outputs: Dict[int, Dict[str, Any]],
        execution_context: ExecutionContext,
    ) -> _StepExecutionOutcome:
        attempts = 0
        guidance: List[str] = []
        while True:
            attempts += 1
            manager = execution_context.intermediate_step_manager
            attempt_id = str(uuid4())
            manager.push_intermediate_step(
                IntermediateStepPayload(
                    UUID=attempt_id,
                    event_type=IntermediateStepType.TASK_START,
                    name=f"step_{step.id}",
                    data=StreamEventData(
                        input={
                            "step": step.model_dump(),
                            "context": self._serialise_context_for_event(context_outputs),
                            "guidance": list(guidance),
                            "attempt": attempts,
                        }
                    ),
                )
            )
            attempt_timings: Dict[str, float] = {}
            try:
                student_start = time.perf_counter()
                student_result = await self._student.aexecute_step(step, context_outputs, guidance)
                attempt_timings["student_ms"] = self._elapsed_ms(student_start)
            except Exception as exc:
                manager.push_intermediate_step(
                    IntermediateStepPayload(
                        UUID=attempt_id,
                        event_type=IntermediateStepType.TASK_END,
                        name=f"step_{step.id}",
                        data=StreamEventData(output={"error": str(exc)}),
                    )
                )
                raise

            step_meta = execution_context.metadata.get("steps", {}).get(step.id, {})
            prior_guidance = list(step_meta.get("guidance", []))

            structured_output = self._obtain_structured_output(student_result)
            status = structured_output.get("status") or student_result.status
            artifacts = student_result.artifacts
            deliverable = student_result.deliverable

            validation_start = time.perf_counter()
            validation = await self._teacher.avalidate_step(
                step,
                student_result.trace,
                structured_output,
                context_outputs,
                prior_guidance,
                guidance,
            )
            attempt_timings["validation_ms"] = self._elapsed_ms(validation_start)
            validation_valid = bool(validation.get("valid"))

        reward_override_payload = None
        if isinstance(validation, dict):
            reward_override_payload = validation.pop("certification_reward", None)

        reward_skipped = True
        reward: AtlasRewardBreakdown
        if reward_override_payload is not None or (validation_valid and status == "ok"):
            judge_context = JudgeContext(
                task=task,
                step=step,
                trace=student_result.trace,
                output=student_result.output,
                attempt=attempts,
                prior_results=context_outputs,
                guidance=prior_guidance,
                reward_override=reward_override_payload,
            )
            reward_start = time.perf_counter()
            reward = await self._evaluator.ajudge(judge_context)
            attempt_timings["reward_ms"] = self._elapsed_ms(reward_start)
            reward_skipped = False
        else:
            reason = "validation_failed" if not validation_valid else f"status_{status}"
            reward = self._build_placeholder_reward(reason)

            evaluation = StepEvaluation(validation=validation, reward=reward)
            should_retry = self._should_retry(status, validation, reward, attempts)

            guidance_text = validation.get("guidance") if isinstance(validation, dict) else None
            if should_retry:
                guidance_message = guidance_text if isinstance(guidance_text, str) else ""
                attempt_timings["guidance_ms"] = 0.0
                if guidance_message:
                    execution_context.append_guidance(step.id, guidance_message)
                    guidance.append(guidance_message)

            total_elapsed = sum(attempt_timings.values())
            attempt_timings["total_ms"] = round(total_elapsed, 3)

            augmented_metadata = self._augment_step_metadata(
                student_result.metadata,
                structured_output,
                attempt_timings,
                reward_skipped,
            )
            student_result.metadata = augmented_metadata

            execution_context.register_step_attempt(
                step.id,
                attempts,
                evaluation,
                timings=attempt_timings,
                reward_skipped=reward_skipped,
                status=status,
            )

            context_entry = None
            if not reward_skipped:
                context_entry = self._build_context_entry(structured_output, student_result.output)

            event_output = {
                "trace": student_result.trace,
                "output": structured_output,
                "evaluation": evaluation.to_dict(),
                "metadata": augmented_metadata,
                "runtime": {
                    "reward_skipped": reward_skipped,
                    "timings_ms": attempt_timings,
                },
                "status": status,
                "artifacts": self._ensure_jsonable(artifacts),
                "deliverable": self._ensure_jsonable(deliverable),
            }
            if context_entry is not None:
                event_output["context_entry"] = context_entry

            manager.push_intermediate_step(
                IntermediateStepPayload(
                    UUID=attempt_id,
                    event_type=IntermediateStepType.TASK_END,
                    name=f"step_{step.id}",
                    data=StreamEventData(output=event_output),
                )
            )

            if not should_retry:
                return _StepExecutionOutcome(
                    result=student_result,
                    evaluation=evaluation,
                    attempts=attempts,
                    context_entry=context_entry,
                    reward_skipped=reward_skipped,
                    status=status,
                    artifacts=artifacts,
                    deliverable=deliverable,
                )

    async def _apply_adaptive_mode(
        self,
        task: str,
        context: ExecutionContext,
        reviewed_plan: Plan,
        dossier: TriageDossier,
    ) -> Tuple[Plan, AdaptiveModeDecision]:
        decision = await self._determine_adaptive_mode(task, context, dossier)
        mode = decision.mode
        if mode in {"auto", "paired"}:
            final_plan = self._convert_to_single_shot_plan(task, reviewed_plan)
        elif mode == "coach":
            final_plan = self._ensure_stepwise_plan(reviewed_plan, context)
        elif mode == "escalate":
            final_plan = self._ensure_stepwise_plan(reviewed_plan, context)
        else:
            final_plan = reviewed_plan
        self._current_retry_limit = self._retry_limit_for_mode(mode)
        self._store_mode_metadata(context, decision)
        return final_plan, decision

    async def _determine_adaptive_mode(
        self,
        task: str,
        context: ExecutionContext,
        dossier: TriageDossier,
    ) -> AdaptiveModeDecision:
        adaptive_cfg = self._adaptive
        if not adaptive_cfg.enabled:
            return AdaptiveModeDecision(mode="escalate", reason="adaptive_disabled")

        if adaptive_cfg.mode_override:
            return AdaptiveModeDecision(mode=adaptive_cfg.mode_override, confidence=1.0, reason="override")

        fingerprint = context.metadata.get("persona_fingerprint")
        certification_needed = (
            adaptive_cfg.certify_first_run
            and isinstance(fingerprint, str)
            and fingerprint not in self._seen_fingerprints
        )
        if certification_needed:
            self._seen_fingerprints.add(fingerprint)
            decision = AdaptiveModeDecision(
                mode="paired",
                confidence=None,
                reason="certification_required",
                certification=True,
            )
            return decision

        probe_result = await self._run_capability_probe(task, context, dossier, fingerprint)
        if probe_result:
            mode = probe_result.mode
            confidence = probe_result.confidence
            if mode not in {"auto", "paired", "coach", "escalate"}:
                mode = self._map_confidence_to_mode(confidence)
            if not mode:
                fallback_mode = adaptive_cfg.probe.fallback_mode
                mode = fallback_mode if fallback_mode in {"coach", "escalate"} else "coach"
            decision = AdaptiveModeDecision(
                mode=mode,
                confidence=confidence,
                reason="probe",
                evidence=list(probe_result.evidence),
                certification=False,
                probe=probe_result,
            )
            if isinstance(fingerprint, str):
                self._seen_fingerprints.add(fingerprint)
            return decision

        fallback_mode = adaptive_cfg.probe.fallback_mode
        if isinstance(fingerprint, str):
            self._seen_fingerprints.add(fingerprint)
        return AdaptiveModeDecision(
            mode=fallback_mode if fallback_mode in {"coach", "escalate"} else "coach",
            reason="probe_unavailable",
        )

    async def _run_capability_probe(
        self,
        task: str,
        context: ExecutionContext,
        dossier: TriageDossier,
        fingerprint: Any,
    ) -> CapabilityProbeResult | None:
        method = getattr(self._teacher, "adaptive_probe", None)
        if not callable(method):
            result = CapabilityProbeResult(
                mode=None,
                confidence=None,
                evidence=["adaptive_probe_unavailable"],
                status="unavailable",
            )
            context.set_capability_probe(
                {
                    "mode": result.mode,
                    "confidence": result.confidence,
                    "evidence": result.evidence,
                    "recommended_personas": result.recommended_personas,
                    "status": result.status,
                    "raw": None,
                }
            )
            return result
        dossier_payload = dossier.model_dump()
        try:
            try:
                payload = await method(
                    task=task,
                    dossier=dossier_payload,
                    fingerprint=fingerprint,
                    execution_metadata=context.metadata,
                )
            except TypeError:
                payload = await method(task, dossier_payload)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("Capability probe execution failed: %s", exc)
            result = CapabilityProbeResult(
                mode=None,
                confidence=None,
                evidence=[f"probe_error:{exc!s}"],
                status="error",
            )
            context.set_capability_probe(
                {
                    "mode": result.mode,
                    "confidence": result.confidence,
                    "evidence": result.evidence,
                    "recommended_personas": result.recommended_personas,
                    "status": result.status,
                    "raw": None,
                }
            )
            return result
        result = self._normalise_probe_payload(payload)
        probe_record = {
            "mode": result.mode,
            "confidence": result.confidence,
            "evidence": result.evidence,
            "recommended_personas": result.recommended_personas,
            "status": result.status,
            "raw": result.raw,
        }
        context.set_capability_probe(probe_record)
        return result

    def _normalise_probe_payload(self, payload: Any) -> CapabilityProbeResult:
        if isinstance(payload, CapabilityProbeResult):
            return payload
        if isinstance(payload, dict):
            mode = payload.get("mode")
            confidence = payload.get("confidence")
            evidence = payload.get("evidence") or payload.get("reasons") or []
            personas = payload.get("recommended_personas") or payload.get("personas") or []
            status = payload.get("status")
            normalized_evidence = [str(item) for item in evidence] if isinstance(evidence, (list, tuple)) else []
            normalized_personas = [str(item) for item in personas] if isinstance(personas, (list, tuple)) else []
            numeric_confidence = None
            if isinstance(confidence, (int, float)):
                numeric_confidence = float(confidence)
            return CapabilityProbeResult(
                mode=str(mode) if isinstance(mode, str) else None,
                confidence=numeric_confidence,
                evidence=normalized_evidence,
                recommended_personas=normalized_personas,
                raw=payload,
                status=str(status) if status is not None else None,
            )
        return CapabilityProbeResult(
            mode=None,
            confidence=None,
            evidence=["invalid_probe_payload"],
            status="invalid",
        )

    def _map_confidence_to_mode(self, confidence: Optional[float]) -> Optional[str]:
        if confidence is None:
            return None
        thresholds = self._adaptive.probe.thresholds
        if confidence >= thresholds.auto:
            return "auto"
        if confidence >= thresholds.paired:
            return "paired"
        if confidence >= thresholds.coach:
            return "coach"
        return "escalate"

    def _retry_limit_for_mode(self, mode: str) -> int:
        if mode == "escalate":
            return self._orchestration.max_retries
        return 0

    def _store_mode_metadata(self, context: ExecutionContext, decision: AdaptiveModeDecision) -> None:
        context.record_mode_decision(
            decision.mode,
            confidence=decision.confidence,
            reason=decision.reason,
            evidence=decision.evidence,
            certification=decision.certification,
        )
        if decision.certification:
            context.mark_certification_run(True)

    def _build_triage_dossier(self, task: str, context: ExecutionContext) -> TriageDossier:
        session_metadata = context.metadata.get("session_metadata")
        metadata_input = session_metadata if isinstance(session_metadata, dict) else {}
        try:
            dossier = self._triage_adapter(task, metadata_input)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("Failed to build triage dossier: %s", exc)
            dossier = default_build_dossier(task, metadata_input)
        return dossier

    def _convert_to_single_shot_plan(self, task: str, plan: Plan) -> Plan:
        single_step = self._build_single_shot_step(task, plan)
        return Plan(steps=[single_step], execution_mode="single_shot")

    def _ensure_stepwise_plan(self, plan: Plan, context: ExecutionContext) -> Plan:
        if plan.execution_mode == "stepwise":
            return plan
        original_payload = context.metadata.get("original_plan")
        if isinstance(original_payload, dict):
            try:
                return Plan.model_validate(original_payload)
            except Exception:  # pragma: no cover - defensive guard
                pass
        return plan
    def _should_retry(
        self,
        status: str,
        validation: Dict[str, Any],
        reward: AtlasRewardBreakdown,
        attempts: int,
    ) -> bool:
        retry_limit = max(self._current_retry_limit, 0)
        max_attempts = retry_limit + 1
        if attempts > max_attempts:
            return False
        status_value = (status or "").strip().lower()
        if status_value == "skipped":
            return False
        if not validation.get("valid", False):
            return attempts <= retry_limit
        if status_value and status_value not in {"ok", "success", "completed"}:
            return attempts <= retry_limit
        return reward.score < self._rim_retry_threshold and attempts <= retry_limit

    def _determine_levels(self, plan: Plan) -> List[List[int]]:
        graph = DependencyGraph(plan)
        return graph.topological_levels()

    def _build_single_shot_step(self, task: str, plan: Plan) -> Step:
        plan_lines: List[str] = []
        for index, step in enumerate(plan.steps, start=1):
            plan_lines.append(f"{index}. {step.description}")
        description_parts = [
            "Produce the complete answer for the task in a single response.",
            "Ensure the output matches the requested format and includes any necessary reasoning.",
        ]
        if plan_lines:
            description_parts.append("Follow this reviewed plan while responding:")
            description_parts.extend(plan_lines)
        description = "\n".join(description_parts)
        return Step(
            id=1,
            description=description,
            tool=None,
            tool_params=None,
            depends_on=[],
        )

    def _lookup_step(self, plan: Plan, step_id: int) -> Step:
        for step in plan.steps:
            if step.id == step_id:
                return step
        raise ValueError(f"Plan is missing step {step_id}")

    def _build_error_evaluation(self, error: str) -> StepEvaluation:
        reward = AtlasRewardBreakdown(
            score=0.0,
            judges=[],
            rationale="runtime_error",
            raw={"error": error},
        )
        return StepEvaluation(
            validation={"valid": False, "error": error},
            reward=reward,
        )

    def _build_placeholder_reward(self, reason: str) -> AtlasRewardBreakdown:
        return AtlasRewardBreakdown(
            score=0.0,
            judges=[],
            rationale=reason,
            raw={"skipped": True, "reason": reason},
        )

    def _serialise_context_for_event(self, context_outputs: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        return {str(step_id): self._ensure_jsonable(payload) for step_id, payload in context_outputs.items()}

    def _obtain_structured_output(self, result: StudentStepResult) -> Dict[str, Any]:
        stored = result.metadata.get("structured_output") if isinstance(result.metadata, dict) else None
        if isinstance(stored, dict):
            return stored
        try:
            parsed = json.loads(result.output)
        except json.JSONDecodeError as exc:
            raise ValueError("Executor output is not valid JSON.") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Executor output must decode to a JSON object.")
        return parsed

    def _augment_step_metadata(
        self,
        metadata: Dict[str, Any] | None,
        structured_output: Dict[str, Any],
        timings: Dict[str, float],
        reward_skipped: bool,
    ) -> Dict[str, Any]:
        base: Dict[str, Any] = {}
        if metadata:
            base.update(metadata)
        status = structured_output.get("status")
        if status is not None:
            base["status"] = status
        result_payload = structured_output.get("result") or {}
        if not isinstance(result_payload, dict):
            result_payload = {}
        artifacts = result_payload.get("artifacts") or {}
        base["artifacts"] = self._ensure_jsonable(artifacts)
        base["deliverable"] = self._ensure_jsonable(result_payload.get("deliverable"))
        reason = structured_output.get("reason")
        if reason is not None:
            base["reason"] = reason
        base["result"] = self._ensure_jsonable(result_payload)
        base["structured_output"] = self._ensure_jsonable(structured_output)
        runtime_meta = base.get("runtime")
        if not isinstance(runtime_meta, dict):
            runtime_meta = {}
        runtime_meta["reward_skipped"] = reward_skipped
        runtime_meta["timings_ms"] = {key: float(value) for key, value in timings.items()}
        base["runtime"] = runtime_meta
        return self._ensure_jsonable(base)

    def _build_context_entry(
        self,
        structured_output: Dict[str, Any],
        output_text: str,
    ) -> Dict[str, Any]:
        entry: Dict[str, Any] = {
            "output_text": output_text,
            "status": structured_output.get("status"),
            "artifacts": self._ensure_jsonable((structured_output.get("result") or {}).get("artifacts") or {}),
            "deliverable": self._ensure_jsonable((structured_output.get("result") or {}).get("deliverable")),
        }
        reason = structured_output.get("reason")
        if reason:
            entry["reason"] = reason
        entry["structured_output"] = self._ensure_jsonable(structured_output)
        return entry

    def _ensure_jsonable(self, value: Any, depth: int = 0) -> Any:
        if depth > 6:
            return str(value)
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, dict):
            normalised: Dict[str, Any] = {}
            for key, item in value.items():
                normalised[str(key)] = self._ensure_jsonable(item, depth + 1)
            return normalised
        if isinstance(value, (list, tuple, set)):
            return [self._ensure_jsonable(item, depth + 1) for item in value]
        if hasattr(value, "model_dump"):
            try:
                dumped = value.model_dump()
            except Exception:
                return str(value)
            return self._ensure_jsonable(dumped, depth + 1)
        if hasattr(value, "to_dict"):
            try:
                dumped = value.to_dict()
            except Exception:
                return str(value)
            return self._ensure_jsonable(dumped, depth + 1)
        return str(value)

    def _elapsed_ms(self, start: float) -> float:
        return round((time.perf_counter() - start) * 1000, 3)
