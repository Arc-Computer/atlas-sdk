"""Illustrative adapter for wrapping LangGraph / DeepAgents graphs with Atlas.

This version bundles a self-contained demo graph, environment, and agent so
`atlas env init --validate` can run end-to-end without external services. When
you're ready to integrate a real LangGraph workflow, provide the compiled graph
via `--env-arg graph=module:create_graph` or swap the factories to your own
implementations.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Iterable, Optional

from atlas import agent, environment
from atlas.sdk.interfaces import DiscoveryContext, TelemetryEmitterProtocol


class ExampleLangGraph:
    """Minimal stand-in for a compiled LangGraph graph."""

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        observation = payload.get("observation") or {}
        history: Iterable[Any] = observation.get("history") or []
        next_step = len(list(history)) + 1
        return {
            "action": f"inspect-step-{next_step}",
            "submit": next_step >= 2,
            "metadata": {"steps": next_step},
        }


def create_example_graph() -> ExampleLangGraph:
    """Return the demo graph that fuels the validation flow."""

    return ExampleLangGraph()


def _resolve_graph(candidate: Any | None) -> Any:
    """Resolve a graph instance or reference string."""

    if candidate is None:
        return create_example_graph()
    if not isinstance(candidate, str):
        return candidate
    if ":" in candidate:
        module_path, attr = candidate.split(":", 1)
    elif "." in candidate:
        module_path, attr = candidate.rsplit(".", 1)
    else:
        raise ValueError(
            f"Graph reference '{candidate}' must be 'module:callable' or 'module.callable'."
        )
    module = import_module(module_path)
    target = getattr(module, attr)
    return target() if callable(target) else target


@environment
class LangGraphEnvironment:
    """Hands observations to a LangGraph-powered agent."""

    def __init__(self, graph: Any | None = None) -> None:
        self._graph = _resolve_graph(graph)
        self._history: list[Any] = []
        self._task: Optional[str] = None

    def reset(self, task: str | None = None) -> Dict[str, Any]:
        self._task = task or "Investigate the LangGraph workflow."
        self._history.clear()
        return {"task": self._task, "history": list(self._history)}

    def step(
        self,
        action: Dict[str, Any] | Any,
    ) -> tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        submit = False
        payload = action
        if isinstance(action, dict):
            submit = bool(action.get("submit"))
            payload = action.get("action")
        self._history.append(payload)
        observation = {"task": self._task, "history": list(self._history)}
        done = submit or len(self._history) >= 2
        reward = 1.0 if done else 0.0
        info = {"submit": submit, "history_length": len(self._history)}
        if submit and isinstance(payload, str):
            info["final_answer"] = payload
        return observation, reward, done, info

    def close(self) -> None:  # pragma: no cover - no external resources
        return None


@agent
class LangGraphAgentWrapper:
    """Adapter that turns a LangGraph graph into an Atlas-compatible agent."""

    def __init__(self, graph: Any | None = None) -> None:
        self._graph = _resolve_graph(graph)
        self._state: Dict[str, Any] = {}
        self._final_answer: Optional[str] = None

    def plan(
        self,
        task: str,
        observation: Any,
        *,
        emit_event: TelemetryEmitterProtocol | None = None,
    ) -> Dict[str, Any]:
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
        if submit and isinstance(action, str):
            self._final_answer = action
        payload: Dict[str, Any] = {"action": action, "submit": submit}
        metadata = result.get("metadata") if isinstance(result, dict) else None
        if isinstance(metadata, dict) and metadata:
            payload["metadata"] = metadata
        return payload

    def summarize(
        self,
        context: DiscoveryContext,
        *,
        history=None,
        emit_event: TelemetryEmitterProtocol | None = None,
    ) -> str:
        if emit_event:
            emit_event.emit("progress", {"stage": "summarize"})
        if isinstance(self._final_answer, str):
            return self._final_answer
        return str(context.observation)


def create_environment(**kwargs: Any) -> LangGraphEnvironment:
    """Factory consumed by ``atlas env init --env-fn``."""

    graph = kwargs.pop("graph", None)
    return LangGraphEnvironment(graph=graph)


def create_langgraph_agent(graph: Any | None = None, **_: Any) -> LangGraphAgentWrapper:
    """Factory passed to ``atlas env init --agent-fn``."""

    return LangGraphAgentWrapper(graph=graph)


__all__ = [
    "LangGraphAgentWrapper",
    "LangGraphEnvironment",
    "create_environment",
    "create_example_graph",
    "create_langgraph_agent",
]
