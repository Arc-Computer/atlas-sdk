"""Student façade orchestrating plan creation and step execution."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any
from typing import Dict
from typing import List
from typing import Sequence
from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import BaseMessage

from atlas.connectors.registry import AgentAdapter
from atlas.runtime.models.intermediate_step import IntermediateStepPayload
from atlas.runtime.models.intermediate_step import IntermediateStepType
from atlas.runtime.models.intermediate_step import StreamEventData
from atlas.runtime.orchestration.execution_context import ExecutionContext
from atlas.config.models import AdapterConfig
from atlas.config.models import StudentConfig
from atlas.config.models import ToolDefinition
from atlas.connectors.langchain_bridge import BYOABridgeLLM
from atlas.connectors.langchain_bridge import build_bridge
from atlas.runtime.agent_loop.tool_loop import ToolCallAgentGraph
from atlas.runtime.agent_loop.tool_loop import ToolCallAgentGraphState
from atlas.prompts import RewrittenStudentPrompts
from atlas.types import Plan
from atlas.types import Step

logger = logging.getLogger(__name__)


@dataclass
class StudentStepResult:
    trace: str
    output: str
    messages: List[BaseMessage]
    metadata: Dict[str, Any] = field(default_factory=dict)
    attempts: int = 1
    status: str = "ok"
    artifacts: Dict[str, Any] = field(default_factory=dict)
    deliverable: Any | None = None


@dataclass
class _GraphNodeDetails:
    name: str
    kind: str


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
        self._llm_stream_state: Dict[str, Dict[str, Any]] = {}

    def update_prompts(self, student_prompts: RewrittenStudentPrompts) -> None:
        """Refresh student prompts and rebuild execution graph to honour new instructions."""
        if (
            self._prompts.planner == student_prompts.planner
            and self._prompts.executor == student_prompts.executor
            and self._prompts.synthesizer == student_prompts.synthesizer
        ):
            return
        self._prompts = student_prompts
        self._graph = None
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
        context.metadata["_reasoning_origin"] = ("student", "plan")
        manager.push_intermediate_step(
            IntermediateStepPayload(
                UUID=event_id,
                event_type=IntermediateStepType.FUNCTION_START,
                name="plan_creation",
                data=StreamEventData(input={"task": task}),
            )
        )
        context.metadata["active_actor"] = "student"
        prompt = self._compose_planner_prompt(task)
        try:
            response = await self._adapter.ainvoke(prompt, metadata={"mode": "planning"})
            if isinstance(response, (dict, list)):
                payload = response
            else:
                payload = self._parse_json_response(response)
            normalised = self._normalise_plan_payload(payload)
            if not isinstance(normalised, dict) or not normalised.get("steps"):
                normalised = {
                    "steps": [
                        {
                            "id": 1,
                            "description": task,
                            "depends_on": [],
                            "tool": None,
                            "tool_params": None,
                        }
                    ]
                }
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
                event_type=IntermediateStepType.FUNCTION_END,
                name="plan_creation",
                data=StreamEventData(output=plan.model_dump()),
            )
        )
        self._consume_reasoning_metadata("student", "plan")
        return plan

    async def aexecute_step(
        self,
        step: Step,
        context: Dict[int, Any],
        guidance: Sequence[str] | None = None,
        recursion_limit: int = 8,
    ) -> StudentStepResult:
        graph = await self._ensure_graph()
        messages = self._build_execution_messages(step, context, guidance)
        execution_context = ExecutionContext.get()
        execution_context.metadata["active_actor"] = "student"
        execution_context.metadata["_reasoning_origin"] = ("student", "execution")
        state = ToolCallAgentGraphState(messages=messages)
        final_messages: Sequence[BaseMessage] | None = None
        step_manager = execution_context.intermediate_step_manager
        llm_snapshots: List[Any] = []
        llm_text_chunks: List[str] = []
        async for event in graph.astream_events(state, config={"recursion_limit": recursion_limit}, version="v2"):
            self._handle_stream_event(step, event, step_manager)
            chunk_messages = self._extract_messages_from_graph_payload(event.get("data", {}).get("chunk"))
            if chunk_messages:
                final_messages = chunk_messages
            chunk_payload = event.get("data", {}).get("chunk")
            if chunk_payload is not None:
                snapshot = self._serialize_graph_payload(chunk_payload)
                if snapshot:
                    llm_snapshots.append(snapshot)
                chunk_text = self._extract_text_from_chunk(chunk_payload)
                if chunk_text:
                    llm_text_chunks.append(chunk_text)
            output_messages = self._extract_messages_from_graph_payload(event.get("data", {}).get("output"))
            if output_messages:
                final_messages = output_messages
        if final_messages is None:
            final_messages = state.messages
        final_state = ToolCallAgentGraphState(messages=list(final_messages))
        output_message = final_state.messages[-1]
        trace = self._build_trace(final_state.messages)
        metadata = self._extract_reasoning_metadata(final_state.messages)
        for payload in self._consume_reasoning_metadata("student", "execution"):
            if payload:
                reasoning_entries = metadata.setdefault("reasoning", [])
                reasoning_entries.append({"origin": "student", "payload": payload})
        if not metadata.get("reasoning"):
            if llm_snapshots:
                snapshot_payload: Dict[str, Any] = {
                    "stream_snapshot": llm_snapshots[-1],
                    "output_text": str(output_message.content),
                }
                combined_text = "".join(llm_text_chunks).strip()
                if combined_text:
                    snapshot_payload["text"] = combined_text
                metadata.setdefault("reasoning", []).append(
                    {
                        "origin": "student",
                        "payload": snapshot_payload,
                    }
                )
            else:
                metadata.setdefault("reasoning", []).append(
                    {
                        "origin": "student",
                        "payload": {
                            "content": str(output_message.content),
                        },
                    }
                )
        structured_output = self._normalise_executor_message(output_message)
        artifacts = structured_output.get("artifacts") or {}
        if not isinstance(artifacts, dict):
            artifacts = {}
        deliverable = structured_output.get("deliverable")
        guidance_reason = structured_output.get("reason")
        raw_text = structured_output.get("text")
        result_payload = structured_output.get("result", {})
        if not isinstance(result_payload, dict):
            result_payload = {"deliverable": deliverable, "artifacts": artifacts}
        metadata["artifacts"] = artifacts
        metadata["deliverable"] = deliverable
        metadata["result"] = result_payload
        metadata["structured_output"] = structured_output
        if guidance_reason is not None:
            metadata["reason"] = guidance_reason
        if raw_text is not None:
            metadata["text"] = raw_text
        output_payload = raw_text if isinstance(raw_text, str) else json.dumps(structured_output, ensure_ascii=False)
        return StudentStepResult(
            trace=trace,
            output=output_payload,
            messages=final_state.messages,
            metadata=metadata,
            status=(structured_output.get("status") or "unknown"),
            artifacts=artifacts,
            deliverable=deliverable,
        )

    async def asynthesize_final_answer(self, task: str, step_results: List[Dict[str, Any]]) -> str:
        context = ExecutionContext.get()
        manager = context.intermediate_step_manager
        event_id = str(uuid4())
        manager.push_intermediate_step(
            IntermediateStepPayload(
                UUID=event_id,
                event_type=IntermediateStepType.FUNCTION_START,
                name="final_synthesis",
                data=StreamEventData(input={"task": task, "step_results": step_results}),
            )
        )
        prompt = self._compose_synthesis_prompt(task, step_results)
        try:
            response = await self._adapter.ainvoke(prompt, metadata={"mode": "synthesis"})
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
                event_type=IntermediateStepType.FUNCTION_END,
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
        context: Dict[int, Any],
        guidance: Sequence[str] | None = None,
        recursion_limit: int = 8,
    ) -> StudentStepResult:
        return self._run_async(self.aexecute_step(step, context, guidance, recursion_limit))

    def synthesize_final_answer(self, task: str, step_results: List[Dict[str, Any]]) -> str:
        return self._run_async(self.asynthesize_final_answer(task, step_results))

    def _compose_planner_prompt(self, task: str) -> str:
        json_direction = (
            "Respond ONLY with JSON matching this schema:\n"
            "{\n"
            "  \"steps\": [\n"
            "    {\n"
            "      \"id\": integer,\n"
            "      \"description\": string,\n"
            "      \"tool\": string | null,\n"
            "      \"tool_params\": object | null,\n"
            "      \"depends_on\": [integer]\n"
            "    }\n"
            "  ]\n"
            "}\n"
            "Do not include any prose before or after the JSON object."
        )
        return "\n\n".join([
            self._prompts.planner,
            f"Task: {task.strip()}",
            json_direction,
        ])

    def _compose_synthesis_prompt(self, task: str, step_results: List[Dict[str, Any]]) -> str:
        serialized_results = json.dumps(step_results, ensure_ascii=False, indent=2)
        execution_mode = ExecutionContext.get().metadata.get("execution_mode", "stepwise")
        return "\n\n".join([
            self._prompts.synthesizer,
            f"Original Task: {task.strip()}",
            f"Execution Mode: {execution_mode}",
            f"Completed Steps: {serialized_results}",
        ])

    def _build_execution_messages(
        self,
        step: Step,
        context: Dict[int, Any],
        guidance: Sequence[str] | None,
    ) -> List[BaseMessage]:
        context_block = json.dumps(context, ensure_ascii=False, indent=2)
        guidance_block = json.dumps(list(guidance or []), ensure_ascii=False, indent=2)
        task_text = ""
        try:
            execution_context = ExecutionContext.get()
            stored_task = execution_context.metadata.get("task")
            if isinstance(stored_task, str):
                task_text = stored_task.strip()
        except Exception:  # pragma: no cover - defensive
            task_text = ""
        payload = [
            f"Step ID: {step.id}",
            f"Description: {step.description}",
        ]
        if task_text:
            payload.append("Original Task:")
            payload.append(task_text)
        payload.extend([
            f"Tool: {step.tool or 'none'}",
            f"Tool Parameters: {json.dumps(step.tool_params or {}, ensure_ascii=False)}",
            f"Dependencies: {step.depends_on}",
            f"Validated Prior Results (artifacts when available): {context_block}",
            f"Guidance History: {guidance_block}",
        ])
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
            if isinstance(message, AIMessage):
                reasoning_payload = self._collect_reasoning_payload(message)
                if reasoning_payload:
                    parts.append(f"{label}_REASONING: {json.dumps(reasoning_payload, ensure_ascii=False)}")
        return "\n".join(parts)

    def _collect_reasoning_payload(self, message: AIMessage) -> Dict[str, Any] | None:
        payload: Dict[str, Any] = {}
        reasoning_keys = (
            "reasoning_content",
            "reasoning",
            "thinking",
            "thinking_blocks",
            "chain_of_thought",
        )
        for key in reasoning_keys:
            value = message.additional_kwargs.get(key) if hasattr(message, "additional_kwargs") else None
            if value:
                payload[key] = value
        return payload or None

    def _extract_text_from_chunk(self, chunk: Any) -> str | None:
        if chunk is None:
            return None
        if isinstance(chunk, str):
            return chunk
        if isinstance(chunk, dict):
            text_value = chunk.get("text")
            if isinstance(text_value, str):
                return text_value
            content_value = chunk.get("content")
            if isinstance(content_value, str):
                return content_value
        return None

    def _parse_json_response(self, text: Any) -> Any:
        if isinstance(text, (dict, list)):
            return text
        if not isinstance(text, str):
            return text
        cleaned = text.strip()
        if cleaned.startswith("```"):
            stripped = cleaned[3:]
            if stripped.lstrip().startswith("json"):
                stripped = stripped.lstrip()[4:]
            cleaned = stripped.strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse JSON response: {cleaned[:200]}") from exc

    def _normalise_executor_message(self, message: BaseMessage) -> Dict[str, Any]:
        raw_text = self._coerce_message_text(message)
        parsed: Dict[str, Any] | None = None
        if isinstance(raw_text, str):
            parsed = self._maybe_parse_executor_json(raw_text)
        status: str | None = None
        deliverable: Any = raw_text
        artifacts: Dict[str, Any] = {}
        reason: str | None = None
        result_payload: Dict[str, Any] = {}
        if isinstance(parsed, dict):
            candidate_status = parsed.get("status")
            if isinstance(candidate_status, str):
                cleaned = candidate_status.strip()
                status = cleaned if cleaned else None
            candidate_result = parsed.get("result")
            if isinstance(candidate_result, dict):
                result_payload = candidate_result
                if "deliverable" in candidate_result:
                    deliverable = candidate_result.get("deliverable")
                artifacts_candidate = candidate_result.get("artifacts")
                if isinstance(artifacts_candidate, dict):
                    artifacts = artifacts_candidate
            candidate_reason = parsed.get("reason")
            if isinstance(candidate_reason, str):
                reason = candidate_reason
        if not result_payload:
            result_payload = {"deliverable": deliverable, "artifacts": artifacts}
        normalised = {
            "deliverable": deliverable,
            "artifacts": artifacts,
            "result": result_payload,
            "text": raw_text,
        }
        if status is not None:
            normalised["status"] = status
        if reason is not None:
            normalised["reason"] = reason
        return normalised

    def _coerce_message_text(self, message: BaseMessage) -> str:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                    else:
                        parts.append(json.dumps(item, ensure_ascii=False))
                else:
                    parts.append(str(item))
            return "".join(parts).strip()
        if isinstance(content, dict):
            return json.dumps(content, ensure_ascii=False)
        return str(content)

    def _maybe_parse_executor_json(self, text: str) -> Dict[str, Any] | None:
        cleaned = text.strip()
        if not cleaned:
            return None
        if cleaned.startswith("```"):
            stripped = cleaned[3:]
            if stripped.lstrip().startswith("json"):
                stripped = stripped.lstrip()[4:]
            cleaned = stripped.strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()
        if not cleaned.startswith("{"):
            return None
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
        return None

    def _consume_reasoning_metadata(self, actor: str, stage: str) -> List[Dict[str, Any]]:
        context = ExecutionContext.get()
        queue = context.metadata.get("_llm_reasoning_queue", [])
        matched: List[Dict[str, Any]] = []
        remaining: List[Dict[str, Any]] = []
        for entry in queue:
            origin = entry.get("origin")
            if origin == (actor, stage):
                matched.append(entry.get("payload") or {})
            else:
                remaining.append(entry)
        context.metadata["_llm_reasoning_queue"] = remaining
        return matched

    def _classify_event_node(self, event: Dict[str, Any]) -> _GraphNodeDetails:
        metadata = event.get("metadata") or {}
        event_name = str(event.get("name") or "graph")
        raw_node = str(metadata.get("langgraph_node") or event_name)
        node_lower = raw_node.lower()
        langchain_type = str(metadata.get("langchain_type") or "").lower()
        event_type = str(event.get("event") or "")
        if event_type.startswith("on_chat_model"):
            return _GraphNodeDetails(name=raw_node, kind="llm")
        if event_type.startswith("on_tool"):
            return _GraphNodeDetails(name=raw_node, kind="tool")
        if "tool" in node_lower or langchain_type in {"tool", "toolkit"}:
            return _GraphNodeDetails(name=raw_node, kind="tool")
        if "agent" in node_lower:
            return _GraphNodeDetails(name=raw_node, kind="task")
        if any(keyword in node_lower for keyword in ("llm", "model", "chat")) or langchain_type in {"llm", "chat_model"}:
            return _GraphNodeDetails(name=raw_node, kind="llm")
        return _GraphNodeDetails(name=raw_node, kind="span")

    def _build_event_metadata(
        self,
        step: Step,
        event: Dict[str, Any],
        node: _GraphNodeDetails,
        run_id: str,
    ) -> Dict[str, Any]:
        actor = ExecutionContext.get().metadata.get("active_actor", "student")
        metadata: Dict[str, Any] = {
            "actor": actor,
            "step_id": step.id,
            "run_id": run_id,
            "node": node.name,
            "node_type": node.kind,
        }
        if tags := event.get("tags"):
            metadata["tags"] = list(tags)
        langgraph_meta = self._filter_langgraph_metadata(event.get("metadata") or {})
        if langgraph_meta:
            metadata["langgraph"] = langgraph_meta
        return metadata

    def _filter_langgraph_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any] | None:
        if not metadata:
            return None
        allowed_keys = {
            "langgraph_node",
            "langgraph_step",
            "langgraph_task_idx",
            "langgraph_triggers",
            "checkpoint_id",
            "checkpoint_ns",
            "ls_provider",
            "ls_model_name",
            "ls_model_type",
            "ls_temperature",
        }
        filtered = {key: metadata[key] for key in allowed_keys if key in metadata}
        return self._jsonify(filtered) if filtered else None

    def _jsonify(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): self._jsonify(sub_value) for key, sub_value in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._jsonify(item) for item in value]
        if hasattr(value, "model_dump"):
            return self._jsonify(value.model_dump())
        if hasattr(value, "__dict__"):
            return self._jsonify(vars(value))
        return str(value)

    def _ensure_llm_stream_state(self, run_id: str) -> Dict[str, Any]:
        return self._llm_stream_state.setdefault(run_id, {"token_count": 0, "chunks": []})

    def _approximate_token_count(self, text: str) -> int:
        stripped = text.strip()
        if not stripped:
            return 0
        return max(1, len(stripped.split()))

    def _normalise_llm_chunk(self, run_id: str, chunk: Any) -> Any:
        if chunk is None:
            return None
        state = self._ensure_llm_stream_state(run_id)
        text: str | None = None
        if isinstance(chunk, dict):
            content = chunk.get("content")
            if isinstance(content, list):
                text = "".join(str(item) for item in content if isinstance(item, str))
            elif isinstance(content, str):
                text = content
            elif "text" in chunk and isinstance(chunk["text"], str):
                text = chunk["text"]
        elif isinstance(chunk, str):
            text = chunk
        if text is None:
            return chunk
        tokens = self._approximate_token_count(text)
        state["token_count"] = state.get("token_count", 0) + tokens
        if text.strip():
            state.setdefault("chunks", []).append(text)
        chunk_payload = dict(chunk if isinstance(chunk, dict) else {})
        chunk_payload["text"] = text
        chunk_payload["token_counts"] = {
            "incremental": tokens,
            "accumulated": state["token_count"],
        }
        return chunk_payload

    def _handle_stream_event(self, step: Step, event: Dict[str, Any], manager: Any | None = None) -> None:
        manager = manager or ExecutionContext.get().intermediate_step_manager
        event_type = event.get("event")
        if not event_type:
            return
        run_id = str(event.get("run_id") or uuid4())
        node = self._classify_event_node(event)
        metadata = self._build_event_metadata(step, event, node, run_id)
        data = event.get("data") or {}
        serialized_input = self._serialize_graph_payload(data.get("input"))
        serialized_output = self._serialize_graph_payload(data.get("output"))
        serialized_chunk = self._serialize_graph_payload(data.get("chunk"))

        handled = False

        if event_type in {"on_chain_start", "on_chat_model_start"}:
            if node.kind == "task":
                manager.push_intermediate_step(
                    IntermediateStepPayload(
                        UUID=run_id,
                        event_type=IntermediateStepType.TASK_START,
                        name=node.name,
                        data=StreamEventData(input=serialized_input),
                        metadata=metadata,
                    )
                )
                handled = True
            elif node.kind == "tool":
                manager.push_intermediate_step(
                    IntermediateStepPayload(
                        UUID=run_id,
                        event_type=IntermediateStepType.TOOL_START,
                        name=node.name,
                        data=StreamEventData(input=serialized_input),
                        metadata=metadata,
                    )
                )
                handled = True
            elif node.kind == "llm":
                self._ensure_llm_stream_state(run_id)
                manager.push_intermediate_step(
                    IntermediateStepPayload(
                        UUID=run_id,
                        event_type=IntermediateStepType.LLM_START,
                        name=node.name,
                        data=StreamEventData(input=serialized_input),
                        metadata=metadata,
                    )
                )
                handled = True

        elif event_type in {"on_chain_end", "on_chat_model_end"}:
            if node.kind == "task":
                manager.push_intermediate_step(
                    IntermediateStepPayload(
                        UUID=run_id,
                        event_type=IntermediateStepType.TASK_END,
                        name=node.name,
                        data=StreamEventData(input=serialized_input, output=serialized_output),
                        metadata=metadata,
                    )
                )
                handled = True
            elif node.kind == "tool":
                manager.push_intermediate_step(
                    IntermediateStepPayload(
                        UUID=run_id,
                        event_type=IntermediateStepType.TOOL_END,
                        name=node.name,
                        data=StreamEventData(input=serialized_input, output=serialized_output),
                        metadata=metadata,
                    )
                )
                handled = True
            elif node.kind == "llm":
                state = self._llm_stream_state.pop(run_id, None)
                if state:
                    metadata = dict(metadata)
                    metadata["token_counts"] = {
                        "approx_total": state.get("token_count", 0),
                        "chunks": len(state.get("chunks", [])),
                    }
                    if state.get("chunks"):
                        context = ExecutionContext.get()
                        queue = context.metadata.setdefault("_llm_reasoning_queue", [])
                        snapshot_text = "".join(state.get("chunks", [])).strip()
                        queue.append(
                            {
                                "origin": ("student", "execution"),
                                "payload": {
                                    "chunks": list(state["chunks"]),
                                    "token_counts": metadata["token_counts"],
                                    "text": snapshot_text or None,
                                },
                            }
                        )
                manager.push_intermediate_step(
                    IntermediateStepPayload(
                        UUID=run_id,
                        event_type=IntermediateStepType.LLM_END,
                        name=node.name,
                        data=StreamEventData(input=serialized_input, output=serialized_output),
                        metadata=metadata,
                    )
                )
                handled = True

        elif event_type in {"on_chain_stream", "on_chat_model_stream"} and node.kind == "llm":
            chunk_payload = self._normalise_llm_chunk(run_id, serialized_chunk)
            manager.push_intermediate_step(
                IntermediateStepPayload(
                    UUID=run_id,
                    event_type=IntermediateStepType.LLM_NEW_TOKEN,
                    name=node.name,
                    data=StreamEventData(chunk=chunk_payload),
                    metadata=metadata,
                )
            )
            handled = True

        if handled:
            return

        # Fallback to generic span events for unclassified telemetry.
        event_name = event.get("name") or node.name
        if event_type == "on_chain_start":
            manager.push_intermediate_step(
                IntermediateStepPayload(
                    UUID=run_id,
                    event_type=IntermediateStepType.SPAN_START,
                    name=event_name,
                    data=StreamEventData(input=serialized_input),
                    metadata=metadata,
                )
            )
        elif event_type == "on_chain_end":
            manager.push_intermediate_step(
                IntermediateStepPayload(
                    UUID=run_id,
                    event_type=IntermediateStepType.SPAN_END,
                    name=event_name,
                    data=StreamEventData(input=serialized_input, output=serialized_output),
                    metadata=metadata,
                )
            )
        elif event_type == "on_chain_stream":
            manager.push_intermediate_step(
                IntermediateStepPayload(
                    UUID=run_id,
                    event_type=IntermediateStepType.SPAN_CHUNK,
                    name=event_name,
                    data=StreamEventData(chunk=serialized_chunk),
                    metadata=metadata,
                )
            )

    def _extract_messages_from_graph_payload(self, payload: Any) -> List[BaseMessage] | None:
        if payload is None:
            return None
        if isinstance(payload, ToolCallAgentGraphState):
            return list(payload.messages)
        if hasattr(payload, "messages"):
            return list(getattr(payload, "messages"))
        if isinstance(payload, dict):
            for value in payload.values():
                messages = self._extract_messages_from_graph_payload(value)
                if messages:
                    return messages
        return None

    def _serialize_graph_payload(self, payload: Any) -> Any:
        if payload is None:
            return None
        if isinstance(payload, ToolCallAgentGraphState):
            return {
                "messages": [self._serialize_message(message) for message in payload.messages],
            }
        if isinstance(payload, BaseMessage):
            return self._serialize_message(payload)
        if isinstance(payload, list):
            return [self._serialize_graph_payload(item) for item in payload]
        if isinstance(payload, dict):
            return {key: self._serialize_graph_payload(value) for key, value in payload.items()}
        if hasattr(payload, "model_dump"):
            return payload.model_dump()
        return str(payload)

    def _serialize_message(self, message: BaseMessage) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "type": message.type,
            "content": message.content,
            "additional_kwargs": getattr(message, "additional_kwargs", None),
        }
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            payload["tool_calls"] = [
                {
                    "name": call.name,
                    "args": call.args,
                    "id": call.id,
                    "type": getattr(call, "type", None),
                }
                for call in tool_calls
            ]
        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id:
            payload["tool_call_id"] = tool_call_id
        status = getattr(message, "status", None)
        if status:
            payload["status"] = status
        return payload

    def _extract_reasoning_metadata(self, messages: Sequence[BaseMessage]) -> Dict[str, Any]:
        reasoning_entries: List[Dict[str, Any]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, AIMessage):
                continue
            payload = self._collect_reasoning_payload(message)
            if not payload:
                continue
            reasoning_entries.append(
                {
                    "message_index": index,
                    "role": message.type,
                    "payload": payload,
                }
            )
        if not reasoning_entries:
            return {}
        return {"reasoning": reasoning_entries}

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
            payload = self._parse_json_response(payload)

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
                    if isinstance(step["depends_on"], list):
                        normalised_deps = []
                        for dep in step["depends_on"]:
                            if isinstance(dep, str):
                                dep_stripped = dep.strip().lstrip("step_")
                                try:
                                    normalised_deps.append(int(dep_stripped))
                                except ValueError:
                                    normalised_deps.append(dep)
                            else:
                                normalised_deps.append(dep)
                        step["depends_on"] = normalised_deps
                    if "tool" not in step:
                        step["tool"] = step.pop("tools", None)
                    if "tool_params" not in step:
                        step["tool_params"] = None
                    tool_value = step.get("tool")
                    if isinstance(tool_value, list):
                        step["tool"] = ", ".join(str(item) for item in tool_value)
                    if "id" in step and isinstance(step["id"], str):
                        raw_id = step["id"].strip()
                        for prefix in ("step_", "step", "s"):
                            if raw_id.lower().startswith(prefix):
                                raw_id = raw_id[len(prefix):]
                                break
                        try:
                            step["id"] = int(raw_id)
                        except (ValueError, TypeError):
                            pass
        return payload
