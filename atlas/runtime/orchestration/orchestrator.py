"""Sequential orchestrator coordinating Teacher, Student, and RIM evaluation."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Sequence
from uuid import uuid4

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
from atlas.types import Plan
from atlas.types import Result
from atlas.types import Step
from atlas.types import StepEvaluation
from atlas.types import StepResult


@dataclass
class _StepExecutionOutcome:
    result: StudentStepResult
    evaluation: StepEvaluation
    attempts: int
    context_entry: Dict[str, Any] | None
    reward_skipped: bool


class Orchestrator:
    def __init__(
        self,
        teacher: Teacher,
        student: Student,
        evaluator: Evaluator,
        orchestration_config: OrchestrationConfig,
        rim_config: RIMConfig,
    ) -> None:
        self._teacher = teacher
        self._student = student
        self._evaluator = evaluator
        self._orchestration = orchestration_config
        self._rim_config = rim_config
        self._rim_retry_threshold = getattr(rim_config, "retry_threshold", 0.6)

    async def arun(self, task: str) -> Result:
        context = ExecutionContext.get()
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
        initial_plan = await self._student.acreate_plan(task)
        reviewed_plan = await self._teacher.areview_plan(task, initial_plan)
        context.metadata["task"] = task
        context.metadata["plan"] = reviewed_plan.model_dump()
        levels = self._determine_levels(reviewed_plan)
        context_outputs: Dict[int, Dict[str, Any]] = {}
        step_summaries: List[Dict[str, Any]] = []
        step_results: List[StepResult] = []
        for level in levels:
            if len(level) == 1:
                step_id = level[0]
                step = self._lookup_step(reviewed_plan, step_id)
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
                        "output": result.output,
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
                steps = [self._lookup_step(reviewed_plan, step_id) for step_id in level]
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
                            "output": result.output,
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
        return Result(final_answer=final_answer, plan=reviewed_plan, step_results=step_results)

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

            validation_start = time.perf_counter()
            validation = await self._teacher.avalidate_step(step, student_result.trace, student_result.output)
            attempt_timings["validation_ms"] = self._elapsed_ms(validation_start)
            validation_valid = bool(validation.get("valid"))

            reward_skipped = not validation_valid
            reward: AtlasRewardBreakdown
            reward_ms: float | None = None
            step_meta = execution_context.metadata.get("steps", {}).get(step.id, {})
            prior_guidance = list(step_meta.get("guidance", []))
            if validation_valid:
                judge_context = JudgeContext(
                    task=task,
                    step=step,
                    trace=student_result.trace,
                    output=student_result.output,
                    attempt=attempts,
                    prior_results=context_outputs,
                    guidance=prior_guidance,
                )
                reward_start = time.perf_counter()
                reward = await self._evaluator.ajudge(judge_context)
                reward_ms = self._elapsed_ms(reward_start)
                attempt_timings["reward_ms"] = reward_ms
            else:
                reward = self._build_validation_failed_reward()

            cached_data, cache_source = self._extract_structured_data(student_result)
            augmented_metadata = self._augment_step_metadata(
                student_result.metadata,
                cached_data,
                cache_source,
                attempt_timings,
                reward_skipped,
            )
            student_result.metadata = augmented_metadata
            evaluation = StepEvaluation(validation=validation, reward=reward)

            should_retry = self._should_retry(validation, reward, attempts)
            guidance_ms: float | None = None
            if should_retry:
                guidance_start = time.perf_counter()
                guidance_text = await self._teacher.agenerate_guidance(step, evaluation.to_dict())
                guidance_ms = self._elapsed_ms(guidance_start)
                attempt_timings["guidance_ms"] = guidance_ms
                execution_context.append_guidance(step.id, guidance_text)
                guidance.append(guidance_text)

            total_elapsed = sum(attempt_timings.values())
            attempt_timings["total_ms"] = round(total_elapsed, 3)
            execution_context.register_step_attempt(
                step.id,
                attempts,
                evaluation,
                timings=attempt_timings,
                reward_skipped=reward_skipped,
            )

            context_entry = None
            if validation_valid:
                context_entry = self._build_context_entry(student_result, cached_data, cache_source)

            event_output = {
                "trace": student_result.trace,
                "output": student_result.output,
                "evaluation": evaluation.to_dict(),
                "metadata": augmented_metadata,
                "runtime": {
                    "reward_skipped": reward_skipped,
                    "timings_ms": attempt_timings,
                },
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
                )

    def _should_retry(self, validation: Dict[str, Any], reward: AtlasRewardBreakdown, attempts: int) -> bool:
        if attempts > self._orchestration.max_retries + 1:
            return False
        if not validation.get("valid", False):
            return attempts <= self._orchestration.max_retries
        return reward.score < self._rim_retry_threshold and attempts <= self._orchestration.max_retries

    def _determine_levels(self, plan: Plan) -> List[List[int]]:
        graph = DependencyGraph(plan)
        return graph.topological_levels()

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

    def _build_validation_failed_reward(self) -> AtlasRewardBreakdown:
        return AtlasRewardBreakdown(
            score=0.0,
            judges=[],
            rationale="validation_failed",
            raw={"skipped": True, "reason": "teacher_validation_failed"},
        )

    def _serialise_context_for_event(self, context_outputs: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        return {str(step_id): self._ensure_jsonable(payload) for step_id, payload in context_outputs.items()}

    def _extract_structured_data(self, student_result: StudentStepResult) -> tuple[Any | None, str | None]:
        metadata = student_result.metadata or {}
        candidate_keys = (
            "cached_data",
            "structured_data",
            "structured_output",
            "structured",
            "parsed_output",
            "artifacts",
            "cache",
        )
        for key in candidate_keys:
            if key in metadata:
                parsed = self._decode_structured_candidate(metadata[key])
                if parsed is not None:
                    return parsed, f"metadata.{key}"
        parsed_output = self._decode_structured_candidate(student_result.output)
        if parsed_output is not None:
            return parsed_output, "output_json"
        return None, None

    def _decode_structured_candidate(self, value: Any) -> Any | None:
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            return value
        if hasattr(value, "model_dump"):
            try:
                return value.model_dump()
            except Exception:
                return None
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError:
                    return None
                if isinstance(parsed, (dict, list)):
                    return parsed
        return None

    def _augment_step_metadata(
        self,
        metadata: Dict[str, Any] | None,
        cached_data: Any | None,
        cache_source: str | None,
        timings: Dict[str, float],
        reward_skipped: bool,
    ) -> Dict[str, Any]:
        base: Dict[str, Any] = {}
        if metadata:
            base.update(metadata)
        runtime_meta = base.get("runtime")
        if not isinstance(runtime_meta, dict):
            runtime_meta = {}
        runtime_meta["reward_skipped"] = reward_skipped
        runtime_meta["timings_ms"] = {key: float(value) for key, value in timings.items()}
        if cache_source:
            runtime_meta["cache_source"] = cache_source
        base["runtime"] = runtime_meta
        if cached_data is not None:
            base["cached_data"] = self._ensure_jsonable(cached_data)
        return self._ensure_jsonable(base)

    def _build_context_entry(
        self,
        student_result: StudentStepResult,
        cached_data: Any | None,
        cache_source: str | None,
    ) -> Dict[str, Any]:
        entry: Dict[str, Any] = {"output_text": student_result.output}
        if cached_data is not None:
            entry["cached_data"] = self._ensure_jsonable(cached_data)
        if cache_source:
            entry["cache_source"] = cache_source
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
