"""Sequential orchestrator coordinating Teacher, Student, and RIM evaluation."""

from __future__ import annotations

import asyncio
from typing import Any
from typing import Dict
from typing import List
from typing import Sequence
from uuid import uuid4

from atlas.config.models import OrchestrationConfig
from atlas.config.models import RIMConfig
from atlas.data_models.intermediate_step import IntermediateStepPayload
from atlas.data_models.intermediate_step import IntermediateStepType
from atlas.data_models.intermediate_step import StreamEventData
from atlas.orchestration.dependency_graph import DependencyGraph
from atlas.orchestration.execution_context import ExecutionContext
from atlas.reward.evaluator import Evaluator
from atlas.reward.judge import JudgeContext
from atlas.roles.student import Student
from atlas.roles.student import StudentStepResult
from atlas.roles.teacher import Teacher
from atlas.runtime.schema import AtlasRewardBreakdown
from atlas.types import Plan
from atlas.types import Result
from atlas.types import Step
from atlas.types import StepEvaluation
from atlas.types import StepResult


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
        context_outputs: Dict[int, str] = {}
        step_summaries: List[Dict[str, Any]] = []
        step_results: List[StepResult] = []
        for level in levels:
            if len(level) == 1:
                step_id = level[0]
                step = self._lookup_step(reviewed_plan, step_id)
                result, evaluation, attempts = await self._run_step(task, step, context_outputs, context)
                context_outputs[step.id] = result.output
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

                    result, evaluation, attempts = outcome
                    context_outputs[step.id] = result.output
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
        context_outputs: Dict[int, str],
        execution_context: ExecutionContext,
    ) -> tuple[StudentStepResult, StepEvaluation, int]:
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
                            "context": context_outputs,
                            "guidance": list(guidance),
                            "attempt": attempts,
                        }
                    ),
                )
            )
            try:
                student_result = await self._student.aexecute_step(step, context_outputs, guidance)
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
            validation = await self._teacher.avalidate_step(step, student_result.trace, student_result.output)
            step_meta = execution_context.metadata.get("steps", {}).get(step.id, {})
            judge_context = JudgeContext(
                task=task,
                step=step,
                trace=student_result.trace,
                output=student_result.output,
                attempt=attempts,
                prior_results=context_outputs,
                guidance=list(step_meta.get("guidance", [])),
            )
            reward = await self._evaluator.ajudge(judge_context)
            evaluation = StepEvaluation(validation=validation, reward=reward)
            execution_context.register_step_attempt(step.id, attempts, evaluation)
            manager.push_intermediate_step(
                IntermediateStepPayload(
                    UUID=attempt_id,
                    event_type=IntermediateStepType.TASK_END,
                    name=f"step_{step.id}",
                    data=StreamEventData(
                        output={
                            "trace": student_result.trace,
                            "output": student_result.output,
                            "evaluation": evaluation.to_dict(),
                            "metadata": student_result.metadata,
                        }
                    ),
                )
            )
            if not self._should_retry(validation, reward, attempts):
                return student_result, evaluation, attempts
            guidance_text = await self._teacher.agenerate_guidance(step, evaluation.to_dict())
            execution_context.append_guidance(step.id, guidance_text)
            guidance.append(guidance_text)

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
