"""Illustrative adapter for wrapping LangGraph / DeepAgents graphs with Atlas."""

from __future__ import annotations

from typing import Any, Dict

from atlas.sdk.interfaces import DiscoveryContext, TelemetryEmitterProtocol


class LangGraphAgentWrapper:
    """Turns a compiled LangGraph graph into an Atlas-compatible agent."""

    def __init__(self, graph) -> None:
        self._graph = graph
        self._state: Dict[str, Any] = {}

    def plan(self, task: str, observation: Any, *, emit_event: TelemetryEmitterProtocol | None = None) -> Dict[str, Any]:
        self._state = {"task": task, "observation": observation}
        if emit_event:
            emit_event.emit("progress", {"stage": "plan", "task": task})
        return {"execution": "graph", "note": "LangGraph adapter"}

    def act(
        self,
        context: DiscoveryContext,
        *,
        emit_event: TelemetryEmitterProtocol | None = None,
    ) -> Dict[str, Any]:
        if emit_event:
            emit_event.emit("progress", {"stage": "act", "step": context.step_index})
        result = self._graph.invoke({"observation": context.observation, "state": self._state})
        action = result.get("action") if isinstance(result, dict) else result
        submit = bool(result.get("submit")) if isinstance(result, dict) else False
        return {"action": action, "submit": submit}

    def summarize(
        self,
        context: DiscoveryContext,
        *,
        history=None,
        emit_event: TelemetryEmitterProtocol | None = None,
    ) -> str:
        if emit_event:
            emit_event.emit("progress", {"stage": "summarize"})
        return str(context.observation)


def create_langgraph_agent(graph) -> LangGraphAgentWrapper:
    """Factory passed to `atlas env init --agent-fn` when using LangGraph/DeepAgents."""

    return LangGraphAgentWrapper(graph)
