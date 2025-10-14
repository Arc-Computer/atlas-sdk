"""Sequential orchestrator coordinating Teacher, Student, and RIM evaluation."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from uuid import uuid4

from atlas.config.models import AdaptiveTeachingConfig
from atlas.config.models import OrchestrationConfig
from atlas.config.models import RIMConfig
from atlas.runtime.models import IntermediateStepPayload
from atlas.runtime.models import IntermediateStepType
from atlas.runtime.models import StreamEventData
from atlas.runtime.orchestration.dependency_graph import DependencyGraph
from atlas.runtime.orchestration.execution_context import ExecutionContext
from atlas.runtime.adaptive import CapabilityProbeClient, CapabilityProbeDecision
from atlas.evaluation.evaluator import Evaluator, SessionStepRecord, SessionTrajectory
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
class AdaptiveModeDecision:
    mode: str
    confidence: Optional[float] = None
    certification: bool = False
    probe: CapabilityProbeDecision | None = None
    source: str | None = None


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
        capability_probe: CapabilityProbeClient | None = None,
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
        self._capability_probe = capability_probe

    async def arun(self, task: str) -> Result:
        context = ExecutionContext.get()
        context.mark_certification_run(False)
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
        logger.info("Orchestrator: starting triage for task '%s' via adapter %s", task, getattr(self._triage_adapter, "__name__", self._triage_adapter))
        triage_adapter_name = getattr(self._triage_adapter, "__qualname__", getattr(self._triage_adapter, "__name__", str(self._triage_adapter)))
        triage_uuid = str(uuid4())
        manager.push_intermediate_step(
            IntermediateStepPayload(
                UUID=triage_uuid,
                event_type=IntermediateStepType.CUSTOM_START,
                name="triage",
                data=StreamEventData(input={"adapter": triage_adapter_name}),
            )
        )
        dossier = self._build_triage_dossier(task, context)
        context.set_triage_dossier(dossier)
        logger.info(
            "Orchestrator: triage dossier built | summary='%s' | risks=%d | tags=%s",
            dossier.summary,
            len(dossier.risks),
            ", ".join(dossier.tags[:3]),
        )
        manager.push_intermediate_step(
            IntermediateStepPayload(
                UUID=triage_uuid,
                event_type=IntermediateStepType.CUSTOM_END,
                name="triage",
                data=StreamEventData(
                    output={
                        "summary": dossier.summary,
                        "risks": [risk.description for risk in dossier.risks[:3]],
                        "tags": dossier.tags[:5],
                    }
                ),
            )
        )
        context.metadata["task"] = task

        if self._persona_refresh is not None:
            await self._persona_refresh()
        self._hydrate_certified_fingerprints(context)

        decision = await self._determine_adaptive_mode(task, context, dossier)
        mode = decision.mode or "escalate"
        context.metadata["execution_mode"] = mode
        self._store_mode_metadata(context, decision)
        confidence_display = f"{decision.confidence:.2f}" if isinstance(decision.confidence, (int, float)) else "n/a"
        logger.info(
            "Orchestrator: adaptive routing selected mode '%s' (confidence=%s, source=%s)",
            mode,
            confidence_display,
            decision.source or "unknown",
        )
        self._emit_adaptive_route_event(context, decision, reason=decision.source)

        if mode in {"auto", "paired"}:
            plan = self._build_direct_plan(task)
            context.metadata["plan"] = plan.model_dump()
            result = await self._run_single_shot(
                task,
                plan,
                require_validation=(mode == "paired"),
                allow_retry=(mode == "paired"),
            )
            await self._evaluate_session_reward(task, plan, result)
        else:
            initial_plan = await self._student.acreate_plan(task)
            reviewed_plan = await self._teacher.areview_plan(task, initial_plan)
            plan_payload = reviewed_plan.model_dump()
            context.metadata["original_plan"] = plan_payload
            if mode == "coach":
                final_plan = self._convert_to_single_shot_plan(task, reviewed_plan)
                context.metadata["single_shot"] = True
            else:
                final_plan = self._ensure_stepwise_plan(reviewed_plan, context)
                context.metadata.pop("single_shot", None)
            context.metadata["plan"] = final_plan.model_dump()
            if mode == "coach":
                result = await self._run_single_shot(
                    task,
                    final_plan,
                    require_validation=True,
                    allow_retry=True,
                )
                await self._evaluate_session_reward(task, final_plan, result)
            else:
                result = await self._run_stepwise(task, final_plan)
                await self._evaluate_session_reward(task, final_plan, result)

        manager.push_intermediate_step(
            IntermediateStepPayload(
                UUID=orchestration_id,
                event_type=IntermediateStepType.WORKFLOW_END,
                name="orchestration",
                data=StreamEventData(output=result.final_answer),
            )
        )
        return result

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
        *,
        require_validation: bool,
        allow_retry: bool,
    ) -> _StepExecutionOutcome:
        attempts = 0
        guidance: List[str] = []
        steps_store = execution_context.metadata.setdefault("steps", {})
        step_meta = steps_store.setdefault(step.id, {})
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
                logger.info("Orchestrator: executing student step %s attempt %d", step.id, attempts)
                student_start = time.perf_counter()
                student_result = await self._student.aexecute_step(step, context_outputs, guidance)
                attempt_timings["student_ms"] = self._elapsed_ms(student_start)
                logger.info(
                    "Orchestrator: student step %s attempt %d completed in %.2f ms",
                    step.id,
                    attempts,
                    attempt_timings["student_ms"],
                )
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

            prior_guidance = list(step_meta.get("guidance", []))

            structured_output = self._obtain_structured_output(student_result)
            student_status = structured_output.get("status")
            artifacts = student_result.artifacts
            deliverable = student_result.deliverable

            is_final_attempt = allow_retry and attempts > self._orchestration.max_retries

            if require_validation and not is_final_attempt:
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
                logger.info(
                    "Orchestrator: teacher validation for step %s attempt %d=%s (guidance=%s)",
                    step.id,
                    attempts,
                    bool(validation.get("valid")) if isinstance(validation, dict) else "n/a",
                    validation.get("guidance") if isinstance(validation, dict) else None,
                )
                validation_valid = bool(validation.get("valid"))
                reward_override_payload = validation.pop("certification_reward", None) if isinstance(validation, dict) else None
            else:
                validation = {"valid": True, "guidance": None}
                reward_override_payload = None
                attempt_timings["validation_ms"] = 0.0
                validation_valid = True

            reward_skipped = True
            if reward_override_payload is not None:
                reward = self._build_placeholder_reward("teacher_certified")
            elif require_validation and not validation_valid:
                reward = self._build_placeholder_reward("validation_failed")
            else:
                reward = self._build_placeholder_reward("deferred")

            evaluation = StepEvaluation(validation=validation, reward=reward)
            should_retry = bool(allow_retry and require_validation and not validation_valid and attempts <= self._orchestration.max_retries)

            guidance_text = validation.get("guidance") if isinstance(validation, dict) else None
            if should_retry and isinstance(guidance_text, str) and guidance_text.strip():
                guidance_message = guidance_text.strip()
                attempt_timings["guidance_ms"] = 0.0
                execution_context.append_guidance(step.id, guidance_message)
                guidance.append(guidance_message)
            else:
                guidance_message = ""

            total_elapsed = sum(attempt_timings.values())
            attempt_timings["total_ms"] = round(total_elapsed, 3)

            augmented_metadata = self._augment_step_metadata(
                student_result.metadata,
                structured_output,
                attempt_timings,
                reward_skipped,
            )
            student_result.metadata = augmented_metadata
            status_label = "completed"
            if require_validation:
                status_label = "passed" if validation_valid else "failed"
            combined_status = (
                student_status.strip()
                if isinstance(student_status, str) and student_status.strip()
                else status_label
            )

            execution_context.register_step_attempt(
                step.id,
                attempts,
                evaluation,
                timings=attempt_timings,
                reward_skipped=reward_skipped,
                status=combined_status,
            )

            context_entry = self._build_context_entry(structured_output, student_result.output)
            context_entry["status"] = combined_status
            step_meta["context_entry"] = context_entry
            step_meta["status"] = combined_status

            event_output = {
                "trace": student_result.trace,
                "output": structured_output,
                "evaluation": evaluation.to_dict(),
                "metadata": augmented_metadata,
                "runtime": {
                    "reward_skipped": reward_skipped,
                    "timings_ms": attempt_timings,
                },
                "status": combined_status,
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
                    status=combined_status,
                    artifacts=artifacts,
                    deliverable=deliverable,
                )

    async def _determine_adaptive_mode(
        self,
        task: str,
        context: ExecutionContext,
        dossier: TriageDossier,
    ) -> AdaptiveModeDecision:
        adaptive_cfg = self._adaptive
        if not adaptive_cfg.enabled:
            return AdaptiveModeDecision(mode="escalate", source="adaptive_disabled")

        if adaptive_cfg.mode_override:
            return AdaptiveModeDecision(
                mode=adaptive_cfg.mode_override,
                confidence=1.0,
                source="mode_override",
            )

        fingerprint = context.metadata.get("persona_fingerprint")
        certification_needed = (
            adaptive_cfg.certify_first_run
            and isinstance(fingerprint, str)
            and not self._has_certified_fingerprint(fingerprint)
        )
        if certification_needed:
            decision = AdaptiveModeDecision(
                mode="paired",
                confidence=None,
                certification=True,
                source="certification_required",
            )
            return decision

        probe_result = await self._run_capability_probe(task, context, dossier, fingerprint)
        if probe_result is not None:
            mode = probe_result.mode
            confidence = probe_result.confidence
            if mode not in {"auto", "paired", "coach", "escalate"}:
                mode = self._map_confidence_to_mode(confidence)
            if not mode:
                fallback_mode = adaptive_cfg.probe.fallback_mode
                mode = fallback_mode if fallback_mode in {"paired", "coach", "escalate"} else "paired"
            decision = AdaptiveModeDecision(
                mode=mode,
                confidence=confidence,
                certification=False,
                probe=probe_result,
                source="capability_probe",
            )
            if isinstance(fingerprint, str):
                self._record_certification_state(fingerprint, False)
            return decision

        fallback_mode = self._capability_probe.fallback_mode if self._capability_probe else adaptive_cfg.probe.fallback_mode
        if isinstance(fingerprint, str):
            self._record_certification_state(fingerprint, False)
        fallback_source = "probe_disabled" if self._capability_probe is None else "probe_fallback"
        return AdaptiveModeDecision(
            mode=fallback_mode if fallback_mode in {"paired", "coach", "escalate"} else "paired",
            source=fallback_source,
        )

    async def _run_capability_probe(
        self,
        task: str,
        context: ExecutionContext,
        dossier: TriageDossier,
        fingerprint: Any,
    ) -> CapabilityProbeDecision | None:
        manager = context.intermediate_step_manager
        if self._capability_probe is None:
            probe_uuid = str(uuid4())
            logger.info("Orchestrator: capability probe disabled; using fallback mode")
            manager.push_intermediate_step(
                IntermediateStepPayload(
                    UUID=probe_uuid,
                    event_type=IntermediateStepType.CUSTOM_START,
                    name="capability_probe",
                    data=StreamEventData(input={"status": "disabled"}),
                )
            )
            manager.push_intermediate_step(
                IntermediateStepPayload(
                    UUID=probe_uuid,
                    event_type=IntermediateStepType.CUSTOM_END,
                    name="capability_probe",
                    data=StreamEventData(output={"status": "disabled"}),
                )
            )
            return None
        probe_name = type(self._capability_probe).__name__
        probe_uuid = str(uuid4())
        logger.info("Orchestrator: invoking capability probe %s", probe_name)
        manager.push_intermediate_step(
            IntermediateStepPayload(
                UUID=probe_uuid,
                event_type=IntermediateStepType.CUSTOM_START,
                name="capability_probe",
                data=StreamEventData(input={"client": probe_name}),
            )
        )
        try:
            decision = await self._capability_probe.arun(
                task=task,
                dossier=dossier.model_dump(),
                fingerprint=str(fingerprint) if isinstance(fingerprint, str) else None,
                execution_metadata=context.metadata,
            )
        except Exception as exc:
            logger.exception("Capability probe execution failed: %s", exc)
            manager.push_intermediate_step(
                IntermediateStepPayload(
                    UUID=probe_uuid,
                    event_type=IntermediateStepType.CUSTOM_END,
                    name="capability_probe",
                    data=StreamEventData(output={"error": str(exc)}),
                )
            )
            return CapabilityProbeDecision(
                mode=None,
                confidence=None,
                raw={"error": str(exc)},
            )
        context.set_capability_probe({"mode": decision.mode, "confidence": decision.confidence})
        logger.info(
            "Orchestrator: capability probe response mode=%s confidence=%s",
            decision.mode,
            f"{decision.confidence:.2f}" if isinstance(decision.confidence, (int, float)) else "n/a",
        )
        manager.push_intermediate_step(
            IntermediateStepPayload(
                UUID=probe_uuid,
                event_type=IntermediateStepType.CUSTOM_END,
                name="capability_probe",
                data=StreamEventData(
                    output={
                        "mode": decision.mode,
                        "confidence": decision.confidence,
                        "raw": decision.raw,
                    }
                ),
            )
        )
        return decision

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

    def _store_mode_metadata(self, context: ExecutionContext, decision: AdaptiveModeDecision) -> None:
        context.record_mode_decision(
            decision.mode,
            confidence=decision.confidence,
            certification=decision.certification,
        )
        if decision.certification:
            context.mark_certification_run(True)
            fingerprint = context.metadata.get("persona_fingerprint")
            if isinstance(fingerprint, str):
                self._record_certification_state(fingerprint, True)

    def _emit_adaptive_route_event(
        self,
        context: ExecutionContext,
        decision: AdaptiveModeDecision,
        *,
        reason: str | None = None,
    ) -> None:
        manager = context.intermediate_step_manager
        route_uuid = str(uuid4())
        manager.push_intermediate_step(
            IntermediateStepPayload(
                UUID=route_uuid,
                event_type=IntermediateStepType.CUSTOM_START,
                name="adaptive_route",
                data=StreamEventData(input={"mode": decision.mode}),
            )
        )
        payload: Dict[str, Any] = {
            "mode": decision.mode,
            "confidence": decision.confidence,
            "certification": decision.certification,
        }
        if reason:
            payload["reason"] = reason
        if decision.probe is not None:
            payload["probe_mode"] = decision.probe.mode
            payload["probe_confidence"] = decision.probe.confidence
        if decision.probe and decision.probe.raw:
            payload["probe_raw"] = decision.probe.raw
        manager.push_intermediate_step(
            IntermediateStepPayload(
                UUID=route_uuid,
                event_type=IntermediateStepType.CUSTOM_END,
                name="adaptive_route",
                data=StreamEventData(output=payload),
            )
        )

    def _build_triage_dossier(self, task: str, context: ExecutionContext) -> TriageDossier:
        session_metadata = context.metadata.get("session_metadata")
        metadata_input = session_metadata if isinstance(session_metadata, dict) else {}
        try:
            dossier = self._triage_adapter(task, metadata_input)
        except Exception as exc:
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
            except Exception:
                pass
        return plan

    def _hydrate_certified_fingerprints(self, context: ExecutionContext) -> None:
        existing = context.metadata.get("adaptive_certified_fingerprints")
        if isinstance(existing, set):
            return
        certified: Set[str] = set()
        persona_memories = context.metadata.get("persona_memories") if isinstance(context.metadata, dict) else None
        if isinstance(persona_memories, dict):
            for records in persona_memories.values():
                if not isinstance(records, list):
                    continue
                for record in records:
                    fingerprint = record.get("trigger_fingerprint")
                    metadata = record.get("metadata")
                    if isinstance(fingerprint, str) and isinstance(metadata, dict) and metadata.get("certified"):
                        certified.add(fingerprint)
        context.metadata["adaptive_certified_fingerprints"] = certified

    def _has_certified_fingerprint(self, fingerprint: str) -> bool:
        context = ExecutionContext.get()
        certs = context.metadata.get("adaptive_certified_fingerprints")
        if isinstance(certs, set):
            return fingerprint in certs
        if isinstance(certs, list):
            return fingerprint in certs
        return False

    def _record_certification_state(self, fingerprint: str, certified: bool) -> None:
        context = ExecutionContext.get()
        certs = context.metadata.setdefault("adaptive_certified_fingerprints", set())
        if isinstance(certs, set):
            if certified:
                certs.add(fingerprint)
            return
        if isinstance(certs, list):
            if certified and fingerprint not in certs:
                certs.append(fingerprint)
            return
        # migrate legacy/non-iterable cases
        if certified:
            context.metadata["adaptive_certified_fingerprints"] = {fingerprint}

    async def _run_single_shot(
        self,
        task: str,
        plan: Plan,
        *,
        require_validation: bool,
        allow_retry: bool,
    ) -> Result:
        context = ExecutionContext.get()
        context.metadata["single_shot"] = True
        single_step = plan.steps[0]
        context_outputs: Dict[int, Dict[str, Any]] = {}
        step_summaries: List[Dict[str, Any]] = []
        step_results: List[StepResult] = []

        outcome = await self._run_step(
            task,
            single_step,
            context_outputs,
            context,
            require_validation=require_validation,
            allow_retry=allow_retry,
        )
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
                "deliverable": outcome.deliverable or result.metadata.get("deliverable"),
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
        if require_validation:
            organized_results = self._teacher.collect_results(step_summaries)
        else:
            organized_results = sorted(step_summaries, key=lambda item: item.get("step_id", 0))
        context.metadata["single_shot_results"] = organized_results
        deliverable = outcome.deliverable or result.metadata.get("deliverable")
        final_answer = deliverable if isinstance(deliverable, str) and deliverable.strip() else result.output
        return Result(final_answer=final_answer, plan=plan, step_results=step_results)

    async def _run_stepwise(self, task: str, plan: Plan) -> Result:
        context = ExecutionContext.get()
        context.metadata.pop("single_shot", None)
        levels = self._determine_levels(plan)
        context_outputs: Dict[int, Dict[str, Any]] = {}
        step_summaries: List[Dict[str, Any]] = []
        step_results: List[StepResult] = []
        execution_mode = str(context.metadata.get("execution_mode", "stepwise") or "stepwise")

        def _store_outcome(step: Step, outcome: _StepExecutionOutcome) -> None:
            if outcome.context_entry is not None:
                context_outputs[step.id] = outcome.context_entry
            student_result = outcome.result
            evaluation = outcome.evaluation
            attempts = outcome.attempts
            step_summaries.append(
                {
                    "step_id": step.id,
                    "description": step.description,
                    "status": outcome.status,
                    "output": student_result.output,
                    "artifacts": outcome.artifacts,
                    "deliverable": student_result.deliverable,
                    "reason": student_result.metadata.get("reason"),
                    "trace": student_result.trace,
                    "evaluation": evaluation.to_dict(),
                    "metadata": student_result.metadata,
                    "attempts": attempts,
                }
            )
            step_results.append(
                StepResult(
                    step_id=step.id,
                    trace=student_result.trace,
                    output=student_result.output,
                    evaluation=evaluation,
                    attempts=attempts,
                    metadata=student_result.metadata,
                )
            )

        for level in levels:
            if len(level) == 1:
                step_id = level[0]
                step = self._lookup_step(plan, step_id)
                outcome = await self._run_step(
                    task,
                    step,
                    context_outputs,
                    context,
                    require_validation=True,
                    allow_retry=True,
                )
                _store_outcome(step, outcome)
            else:
                steps = [self._lookup_step(plan, step_id) for step_id in level]
                if execution_mode == "escalate":
                    for step in steps:
                        outcome = await self._run_step(
                            task,
                            step,
                            context_outputs,
                            context,
                            require_validation=True,
                            allow_retry=True,
                        )
                        _store_outcome(step, outcome)
                else:
                    tasks = [
                        self._run_step(
                            task,
                            step,
                            dict(context_outputs),
                            context,
                            require_validation=True,
                            allow_retry=True,
                        )
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

                        _store_outcome(step, outcome)
                    if captured_exception is not None:
                        raise captured_exception

        organized_results = self._teacher.collect_results(step_summaries)
        final_answer = await self._student.asynthesize_final_answer(task, organized_results)
        return Result(final_answer=final_answer, plan=plan, step_results=step_results)

    async def _evaluate_session_reward(self, task: str, plan: Plan, result: Result) -> None:
        context = ExecutionContext.get()
        try:
            trajectory = self._build_session_trajectory(task, plan, result)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("Failed to build session trajectory: %s", exc)
            return
        try:
            evaluation = await self._evaluator.aevaluate_session(trajectory)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("Session-level reward evaluation failed: %s", exc)
            return
        context.set_session_reward(
            evaluation.reward,
            student_learning=evaluation.student_learning,
            teacher_learning=evaluation.teacher_learning,
        )

    def _build_session_trajectory(self, task: str, plan: Plan, result: Result) -> SessionTrajectory:
        context = ExecutionContext.get()
        steps_meta = context.metadata.get("steps", {}) if isinstance(context.metadata, dict) else {}
        step_lookup = {step.id: step for step in plan.steps}
        records: List[SessionStepRecord] = []
        guidance_present = False

        for step_result in result.step_results:
            step = step_lookup.get(step_result.step_id)
            if step is None:
                continue
            meta = steps_meta.get(step_result.step_id, {}) or {}
            guidance = meta.get("guidance") if isinstance(meta, dict) else None
            if guidance:
                guidance_present = True
            prior_results = meta.get("context_entry") if isinstance(meta, dict) else None
            status = meta.get("status") if isinstance(meta, dict) else None
            records.append(
                SessionStepRecord(
                    step=step,
                    trace=step_result.trace,
                    output=step_result.output,
                    attempts=step_result.attempts,
                    guidance=list(guidance) if isinstance(guidance, list) else None,
                    status=status,
                    validation=step_result.evaluation.validation,
                    prior_results=prior_results if isinstance(prior_results, dict) else None,
                    metadata=step_result.metadata,
                )
            )

        teacher_intervened = guidance_present or any((record.attempts or 0) > 1 for record in records)
        session_metadata = context.metadata.get("session_metadata") if isinstance(context.metadata, dict) else None
        execution_mode = context.metadata.get("execution_mode") if isinstance(context.metadata, dict) else None

        return SessionTrajectory(
            task=task,
            final_answer=result.final_answer,
            plan=plan.model_dump(),
            steps=records,
            execution_mode=execution_mode,
            teacher_intervened=bool(teacher_intervened),
            session_metadata=session_metadata if isinstance(session_metadata, dict) else None,
            focus_prompt=None,
        )

    def _determine_levels(self, plan: Plan) -> List[List[int]]:
        graph = DependencyGraph(plan)
        return graph.topological_levels()

    def _build_direct_plan(self, task: str) -> Plan:
        base_plan = Plan(steps=[], execution_mode="single_shot")
        single_step = self._build_single_shot_step(task, base_plan)
        return Plan(steps=[single_step], execution_mode="single_shot")

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
