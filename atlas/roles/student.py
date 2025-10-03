"""Student façade orchestrating plan creation and step execution."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Sequence
from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import BaseMessage

from atlas.agent.registry import AgentAdapter
from atlas.data_models.intermediate_step import IntermediateStepPayload
from atlas.data_models.intermediate_step import IntermediateStepType
from atlas.data_models.intermediate_step import StreamEventData
from atlas.orchestration.execution_context import ExecutionContext
from atlas.config.models import AdapterConfig
from atlas.config.models import StudentConfig
from atlas.config.models import ToolDefinition
from atlas.roles.student_bridge import BYOABridgeLLM
from atlas.roles.student_bridge import build_bridge
from atlas.roles.student_core import ToolCallAgentGraph
from atlas.roles.student_core import ToolCallAgentGraphState
from atlas.transition.rewriter import RewrittenStudentPrompts
from atlas.types import Plan
from atlas.types import Step

logger = logging.getLogger(__name__)


@dataclass
class StudentStepResult:
    trace: str
    output: str
    messages: List[BaseMessage]
    attempts: int = 1


class Student:
    def __init__(
        self,
        adapter: AgentAdapter,
        adapter_config: AdapterConfig,
        student_config: StudentConfig,
        student_prompts: RewrittenStudentPrompts,
    ) -> None:
        self._adapter = adapter
        self._student_config = student_config
        self._prompts: RewrittenStudentPrompts = student_prompts
        self._bridge_llm, self._tools = build_bridge(adapter, adapter_config.tools)
        self._graph: Any | None = None
        self._graph_builder = ToolCallAgentGraph(
            llm=self._bridge_llm,
            tools=self._tools,
            system_prompt=self._prompts.executor,
            callbacks=None,
            detailed_logs=False,
            log_response_max_chars=1000,
            handle_tool_errors=True,
            return_direct=None,
        )

    async def acreate_plan(self, task: str) -> Plan:
        context = ExecutionContext.get()
        manager = context.intermediate_step_manager
        event_id = str(uuid4())
        manager.push_intermediate_step(
            IntermediateStepPayload(
                UUID=event_id,
                event_type=IntermediateStepType.WORKFLOW_START,
                name="plan_creation",
                data=StreamEventData(input={"task": task}),
            )
        )
        prompt = self._compose_planner_prompt(task)
        try:
            response = await self._adapter.ainvoke(prompt, metadata={"mode": "planning", "task": task})
            if isinstance(response, (dict, list)):
                payload = response
            else:
                payload = json.loads(response)
            normalised = self._normalise_plan_payload(payload)
            plan = Plan.model_validate(normalised)
        except Exception as exc:
            manager.push_intermediate_step(
                IntermediateStepPayload(
                    UUID=event_id,
                    event_type=IntermediateStepType.WORKFLOW_END,
                    name="plan_creation",
                    data=StreamEventData(output={"error": str(exc)}),
                )
            )
            raise
        manager.push_intermediate_step(
            IntermediateStepPayload(
                UUID=event_id,
                event_type=IntermediateStepType.WORKFLOW_END,
                name="plan_creation",
                data=StreamEventData(output=plan.model_dump()),
            )
        )
        return plan

    async def aexecute_step(
        self,
        step: Step,
        context: Dict[int, str],
        guidance: Sequence[str] | None = None,
        recursion_limit: int = 8,
    ) -> StudentStepResult:
        graph = await self._ensure_graph()
        messages = self._build_execution_messages(step, context, guidance)
        state = ToolCallAgentGraphState(messages=messages)
        result_state = await graph.ainvoke(state, config={"recursion_limit": recursion_limit})
        final_state = ToolCallAgentGraphState(**result_state)
        output_message = final_state.messages[-1]
        trace = self._build_trace(final_state.messages)
        return StudentStepResult(trace=trace, output=str(output_message.content), messages=final_state.messages)

    async def asynthesize_final_answer(self, task: str, step_results: List[Dict[str, Any]]) -> str:
        context = ExecutionContext.get()
        manager = context.intermediate_step_manager
        event_id = str(uuid4())
        manager.push_intermediate_step(
            IntermediateStepPayload(
                UUID=event_id,
                event_type=IntermediateStepType.WORKFLOW_START,
                name="final_synthesis",
                data=StreamEventData(input={"task": task, "step_results": step_results}),
            )
        )
        prompt = self._compose_synthesis_prompt(task, step_results)
        try:
            response = await self._adapter.ainvoke(prompt, metadata={"mode": "synthesis", "task": task})
            if isinstance(response, str):
                final_answer = response
            else:
                final_answer = json.dumps(response)
        except Exception as exc:
            manager.push_intermediate_step(
                IntermediateStepPayload(
                    UUID=event_id,
                    event_type=IntermediateStepType.WORKFLOW_END,
                    name="final_synthesis",
                    data=StreamEventData(output={"error": str(exc)}),
                )
            )
            raise
        manager.push_intermediate_step(
            IntermediateStepPayload(
                UUID=event_id,
                event_type=IntermediateStepType.WORKFLOW_END,
                name="final_synthesis",
                data=StreamEventData(output=final_answer),
            )
        )
        return final_answer

    def create_plan(self, task: str) -> Plan:
        return self._run_async(self.acreate_plan(task))

    def execute_step(
        self,
        step: Step,
        context: Dict[int, str],
        guidance: Sequence[str] | None = None,
        recursion_limit: int = 8,
    ) -> StudentStepResult:
        return self._run_async(self.aexecute_step(step, context, guidance, recursion_limit))

    def synthesize_final_answer(self, task: str, step_results: List[Dict[str, Any]]) -> str:
        return self._run_async(self.asynthesize_final_answer(task, step_results))

    def _compose_planner_prompt(self, task: str) -> str:
        return f"{self._prompts.planner}\n\nTask: {task.strip()}"

    def _compose_synthesis_prompt(self, task: str, step_results: List[Dict[str, Any]]) -> str:
        serialized_results = json.dumps(step_results, ensure_ascii=False, indent=2)
        return "\n\n".join([
            self._prompts.synthesizer,
            f"Original Task: {task.strip()}",
            f"Completed Steps: {serialized_results}",
        ])

    def _build_execution_messages(
        self,
        step: Step,
        context: Dict[int, str],
        guidance: Sequence[str] | None,
    ) -> List[BaseMessage]:
        context_block = json.dumps(context, ensure_ascii=False, indent=2)
        guidance_block = json.dumps(list(guidance or []), ensure_ascii=False, indent=2)
        payload = [
            f"Step ID: {step.id}",
            f"Description: {step.description}",
            f"Tool: {step.tool or 'none'}",
            f"Tool Parameters: {json.dumps(step.tool_params or {}, ensure_ascii=False)}",
            f"Dependencies: {step.depends_on}",
            f"Context: {context_block}",
            f"Guidance: {guidance_block}",
        ]
        user_message = "\n".join(payload)
        return [
            SystemMessage(content=self._prompts.executor),
            HumanMessage(content=user_message),
        ]

    def _build_trace(self, messages: Sequence[BaseMessage]) -> str:
        parts = []
        for message in messages:
            if isinstance(message, SystemMessage):
                continue
            label = message.type.upper()
            content = message.content
            if isinstance(message, AIMessage) and message.tool_calls:
                tool_block = json.dumps([
                    {"name": call.name, "args": call.args, "id": call.id} for call in message.tool_calls
                ], ensure_ascii=False)
                parts.append(f"{label}: tool_calls={tool_block}")
            else:
                parts.append(f"{label}: {content}")
        return "\n".join(parts)

    async def _ensure_graph(self):
        if self._graph is None:
            self._graph = await self._graph_builder.build_graph()
        return self._graph

    def _run_async(self, coroutine):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)
        raise RuntimeError("Student synchronous methods cannot be used inside an active event loop")

    def _normalise_plan_payload(self, payload):
        if isinstance(payload, str):
            payload = json.loads(payload)

        if isinstance(payload, list):
            payload = {"steps": payload}

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
                    if "id" in step and isinstance(step["id"], str):
                        try:
                            step["id"] = int(step["id"].lstrip("s"))
                        except (ValueError, AttributeError):
                            try:
                                step["id"] = int(step["id"])
                            except (ValueError, TypeError):
                                pass
        return payload
