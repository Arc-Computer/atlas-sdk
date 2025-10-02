"""Sequential orchestrator coordinating Teacher, Student, and RIM evaluation."""

from __future__ import annotations

import asyncio
import json
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
from langchain_core.messages import AIMessage
from langchain_core.messages import ToolMessage
from atlas.roles.student import Student
from atlas.roles.student import StudentStepResult
from atlas.roles.teacher import Teacher
from atlas.types import Plan
from atlas.types import Result
from atlas.types import Step
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
        execution_order = self._determine_order(reviewed_plan)
        context_outputs: Dict[int, str] = {}
        step_summaries: List[Dict[str, Any]] = []
        step_results: List[StepResult] = []
        for step_id in execution_order:
            step = self._lookup_step(reviewed_plan, step_id)
            result, evaluation, attempts = await self._run_step(task, step, context_outputs, context)
            context_outputs[step.id] = result.output
            step_payload = {
                "step_id": step.id,
                "description": step.description,
                "output": result.output,
                "trace": result.trace,
                "evaluation": evaluation,
                "attempts": attempts,
            }
            step_summaries.append(step_payload)
            step_results.append(
                StepResult(
                    step_id=step.id,
                    trace=result.trace,
                    output=result.output,
                    evaluation=evaluation,
                    attempts=attempts,
                )
            )
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
    ) -> tuple[StudentStepResult, Dict[str, Any], int]:
        attempts = 0
        guidance: List[str] = []
        evaluation: Dict[str, Any] = {}
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
            self._emit_tool_events(execution_context, student_result.messages)
            validation = await self._teacher.avalidate_step(step, student_result.trace, student_result.output)
            judge_context = JudgeContext(
                task=task,
                step=step,
                trace=student_result.trace,
                output=student_result.output,
                attempt=attempts,
                prior_results=context_outputs,
            )
            reward = await self._evaluator.ajudge(judge_context)
            evaluation = {"validation": validation, "reward": reward}
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
                            "evaluation": evaluation,
                        }
                    ),
                )
            )
            if not self._should_retry(validation, reward["score"], attempts):
                return student_result, evaluation, attempts
            guidance_text = await self._teacher.agenerate_guidance(step, evaluation)
            execution_context.append_guidance(step.id, guidance_text)
            guidance.append(guidance_text)

    def _emit_tool_events(self, execution_context: ExecutionContext, messages: Sequence[Any]) -> None:
        manager = execution_context.intermediate_step_manager
        sessions: Dict[str, str] = {}
        for message in messages:
            if isinstance(message, AIMessage) and message.tool_calls:
                for index, call in enumerate(message.tool_calls):
                    name = getattr(call, "name", None) or call.get("name") if isinstance(call, dict) else None
                    if not name:
                        continue
                    call_id = getattr(call, "id", None) or call.get("id") if isinstance(call, dict) else None
                    if call_id is None:
                        call_id = f"{name}-{index}"
                    arguments = getattr(call, "args", None)
                    if isinstance(call, dict):
                        arguments = call.get("arguments") or call.get("args")
                    arguments = self._normalise_tool_arguments(arguments)
                    tool_uuid = str(uuid4())
                    sessions[call_id] = tool_uuid
                    manager.push_intermediate_step(
                        IntermediateStepPayload(
                            UUID=tool_uuid,
                            event_type=IntermediateStepType.TOOL_START,
                            name=name,
                            data=StreamEventData(input=arguments),
                            metadata={"tool_call_id": call_id},
                        )
                    )
            elif isinstance(message, ToolMessage):
                call_id = message.tool_call_id
                tool_uuid = sessions.get(call_id, str(uuid4()))
                manager.push_intermediate_step(
                    IntermediateStepPayload(
                        UUID=tool_uuid,
                        event_type=IntermediateStepType.TOOL_END,
                        name=call_id or "tool_output",
                        data=StreamEventData(output=self._stringify_tool_content(message.content)),
                        metadata={"tool_call_id": call_id},
                    )
                )

    def _normalise_tool_arguments(self, arguments: Any) -> Any:
        if arguments is None:
            return {}
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
                return parsed
            except json.JSONDecodeError:
                return arguments
        return arguments

    def _stringify_tool_content(self, content: Any) -> str:
        if isinstance(content, (dict, list)):
            return json.dumps(content)
        return str(content)

    def _should_retry(self, validation: Dict[str, Any], score: float, attempts: int) -> bool:
        if attempts > self._orchestration.max_retries + 1:
            return False
        if not validation.get("valid", False):
            return attempts <= self._orchestration.max_retries
        return score < self._rim_config.retry_threshold and attempts <= self._orchestration.max_retries

    def _determine_order(self, plan: Plan) -> List[int]:
        graph = DependencyGraph(plan)
        levels = graph.topological_levels()
        ordered: List[int] = []
        for level in levels:
            if len(level) != 1:
                raise ValueError("Parallel execution is not supported in the sequential orchestrator")
            ordered.append(level[0])
        return ordered

    def _lookup_step(self, plan: Plan, step_id: int) -> Step:
        for step in plan.steps:
            if step.id == step_id:
                return step
        raise ValueError(f"Plan is missing step {step_id}")
