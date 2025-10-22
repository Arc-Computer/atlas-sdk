"""Illustrative adapter for wrapping LangGraph / DeepAgents graphs with Atlas."""

from __future__ import annotations

from typing import Any, Dict

from atlas.sdk.interfaces import DiscoveryContext, TelemetryEmitterProtocol


class LangGraphEnvironment:
    """Minimal environment shim you can customise for your stack."""

    def __init__(self, *, dataset: str = "incidents") -> None:
        self._dataset = dataset
        self._step = 0

    def reset(self, task: str | None = None) -> Dict[str, Any]:
        self._step = 0
        return {"task": task or "Investigate incident", "dataset": self._dataset}

    def step(self, action: Dict[str, Any], submit: bool = False):
        self._step += 1
        done = submit or self._step >= 3
        observation = {
            "step": self._step,
            "action": action,
            "submit": submit,
        }
        reward = 1.0 if done else 0.0
        info = {"dataset": self._dataset}
        return observation, reward, done, info

    def close(self) -> None:  # pragma: no cover - placeholder cleanup
        return None


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


def create_environment(dataset: str = "incidents") -> LangGraphEnvironment:
    """Factory passed to `atlas env init --env-fn` when using LangGraph/DeepAgents."""

    return LangGraphEnvironment(dataset=dataset)


def create_langgraph_agent(graph) -> LangGraphAgentWrapper:
    """Factory passed to `atlas env init --agent-fn` when using LangGraph/DeepAgents."""

    return LangGraphAgentWrapper(graph)
