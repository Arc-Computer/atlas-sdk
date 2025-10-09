"""LangChain callback handler that bridges telemetry into the execution context."""

from __future__ import annotations

import json
from contextvars import ContextVar
from typing import Any, Dict, Iterable, List

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.tracers.context import register_configure_hook

from atlas.runtime.models import IntermediateStepPayload
from atlas.runtime.models import IntermediateStepType
from atlas.runtime.models import StreamEventData
from atlas.runtime.orchestration.execution_context import ExecutionContext


def _serialise_messages(messages: Iterable[BaseMessage]) -> List[Dict[str, Any]]:
    serialised: List[Dict[str, Any]] = []
    for message in messages:
        if isinstance(message, list):
            serialised.extend(_serialise_messages(message))
            continue
        serialised.append(
            {
                "type": message.type,
                "content": message.content,
                "additional_kwargs": getattr(message, "additional_kwargs", None),
            }
        )
    return serialised


class TelemetryCallbackHandler(AsyncCallbackHandler):
    """Capture LangChain events and emit IntermediateStepPayload objects."""

    async def on_chat_model_start(self, serialized, messages, run_id, parent_run_id=None, **kwargs) -> None:
        manager = self._manager()
        if manager is None:
            return
        payload = IntermediateStepPayload(
            UUID=str(run_id),
            event_type=IntermediateStepType.LLM_START,
            name=serialized.get("name") or "chat_model",
            data=StreamEventData(input={"messages": _serialise_messages(messages)}),
            metadata=self._metadata(parent_run_id),
        )
        manager.push_intermediate_step(payload)

    async def on_chat_model_stream(self, output, run_id, parent_run_id=None, **kwargs) -> None:
        manager = self._manager()
        if manager is None:
            return
        text = self._chunk_to_text(output)
        if text is None:
            return
        payload = IntermediateStepPayload(
            UUID=str(run_id),
            event_type=IntermediateStepType.LLM_NEW_TOKEN,
            name="chat_model",
            data=StreamEventData(chunk=text),
            metadata=self._metadata(parent_run_id),
        )
        manager.push_intermediate_step(payload)

    async def on_chat_model_end(self, response, run_id, parent_run_id=None, **kwargs) -> None:
        manager = self._manager()
        if manager is None:
            return
        text = self._chat_response_to_text(response)
        payload = IntermediateStepPayload(
            UUID=str(run_id),
            event_type=IntermediateStepType.LLM_END,
            name="chat_model",
            data=StreamEventData(output=text),
            metadata=self._metadata(parent_run_id),
        )
        manager.push_intermediate_step(payload)

    async def on_tool_start(self, serialized, input_str, run_id, parent_run_id=None, **kwargs) -> None:
        manager = self._manager()
        if manager is None:
            return
        payload = IntermediateStepPayload(
            UUID=str(run_id),
            event_type=IntermediateStepType.TOOL_START,
            name=serialized.get("name") or "tool",
            data=StreamEventData(input=self._safe_json(input_str)),
            metadata=self._metadata(parent_run_id),
        )
        manager.push_intermediate_step(payload)

    async def on_tool_end(self, output, run_id, parent_run_id=None, **kwargs) -> None:
        manager = self._manager()
        if manager is None:
            return
        payload = IntermediateStepPayload(
            UUID=str(run_id),
            event_type=IntermediateStepType.TOOL_END,
            name="tool",
            data=StreamEventData(output=self._safe_json(output)),
            metadata=self._metadata(parent_run_id),
        )
        manager.push_intermediate_step(payload)

    def _metadata(self, parent_run_id: str | None) -> Dict[str, Any]:
        context = ExecutionContext.get()
        actor = context.metadata.get("active_actor", "student")
        metadata: Dict[str, Any] = {"actor": actor}
        if parent_run_id:
            metadata["parent_run_id"] = str(parent_run_id)
        return metadata

    def _manager(self):
        try:
            return ExecutionContext.get().intermediate_step_manager
        except Exception:
            return None

    def _chunk_to_text(self, output: Any) -> str | None:
        if output is None:
            return None
        if isinstance(output, ChatGenerationChunk):
            content = getattr(output, "message", None)
            if content is not None:
                return getattr(content, "content", None)
            return getattr(output, "text", None)
        if hasattr(output, "content"):
            return getattr(output, "content")
        if isinstance(output, dict):
            return json.dumps(output)
        return str(output)

    def _chat_response_to_text(self, response: Any) -> Any:
        if response is None:
            return None
        if hasattr(response, "generations"):
            generations = getattr(response, "generations")
            if generations:
                first = generations[0]
                if hasattr(first, "text"):
                    return first.text
        if hasattr(response, "message"):
            message = getattr(response, "message")
            return getattr(message, "content", None)
        if isinstance(response, dict):
            return response
        return str(response)

    def _safe_json(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        if isinstance(value, (dict, list)):
            return value
        return str(value)


_HANDLER: TelemetryCallbackHandler | None = None
_HANDLER_VAR: ContextVar[TelemetryCallbackHandler | None] = ContextVar(
    "atlas_telemetry_callback",
    default=None,
)


def configure_langchain_callbacks() -> None:
    """Register the telemetry callback with LangChain once per process."""

    global _HANDLER  # noqa: PLW0603
    if _HANDLER is not None:
        return
    _HANDLER = TelemetryCallbackHandler()
    register_configure_hook(_HANDLER_VAR, inheritable=True)
    _HANDLER_VAR.set(_HANDLER)
