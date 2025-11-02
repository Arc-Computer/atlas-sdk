"""Illustrative adapter for wrapping LangGraph / DeepAgents graphs with Atlas.

This version bundles a self-contained demo graph, environment, and agent so
`atlas env init --validate` can run end-to-end without external services. When
you're ready to integrate a real LangGraph workflow, provide the compiled graph
via `--env-arg graph=module:create_graph` or swap the factories to your own
implementations.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
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

    def __init__(self, graph: Any | None = None, credentials: Dict[str, Any] | None = None) -> None:
        self._graph = _resolve_graph(graph)
        self._history: list[Any] = []
        self._task: Optional[str] = None
        self._credentials: Dict[str, Any] | None = _load_credentials(credentials)

    def reset(self, task: str | None = None) -> Dict[str, Any]:
        self._task = task or "Investigate the LangGraph workflow."
        self._history.clear()
        observation: Dict[str, Any] = {"task": self._task, "history": list(self._history)}
        if self._credentials and self._credentials.get("user"):
            observation["identity"] = self._credentials["user"]
        return observation

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
        if self._credentials and self._credentials.get("user"):
            info["identity"] = self._credentials["user"]
        if submit and isinstance(payload, str):
            info["final_answer"] = payload
        return observation, reward, done, info

    def close(self) -> None:  # pragma: no cover - no external resources
        return None


@agent
class LangGraphAgentWrapper:
    """Adapter that turns a LangGraph graph into an Atlas-compatible agent."""

    def __init__(self, graph: Any | None = None, credentials: Dict[str, Any] | None = None) -> None:
        self._graph = _resolve_graph(graph)
        self._state: Dict[str, Any] = {}
        self._final_answer: Optional[str] = None
        self._credentials: Dict[str, Any] | None = _load_credentials(credentials)
        self._thread_id = str(uuid.uuid4())

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
        config: Dict[str, Any] = {"configurable": {}}
        if self._credentials:
            config["configurable"]["_credentials"] = self._credentials
        config["configurable"].setdefault("thread_id", self._thread_id)
        payload = {"observation": context.observation, "state": self._state}
        result = _invoke_graph(self._graph, payload, config=config)
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
    credentials = kwargs.pop("credentials", None)
    return LangGraphEnvironment(graph=graph, credentials=credentials)


def create_langgraph_agent(graph: Any | None = None, **kwargs: Any) -> LangGraphAgentWrapper:
    """Factory passed to ``atlas env init --agent-fn``."""

    credentials = kwargs.pop("credentials", None)
    return LangGraphAgentWrapper(graph=graph, credentials=credentials)


def _invoke_graph(graph: Any, payload: Dict[str, Any], *, config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Invoke a LangGraph synchronously, falling back to async execution when required."""

    if hasattr(graph, "ainvoke"):

        async def _run_async() -> Dict[str, Any]:
            return await graph.ainvoke(payload, config=config)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_run_async())
        else:  # pragma: no cover - not currently exercised in tests
            return loop.run_until_complete(_run_async())

    return graph.invoke(payload, config=config)


def _load_credentials(candidate: Dict[str, Any] | str | None = None) -> Dict[str, Any] | None:
    """Load Auth0 credential scaffolding used by Assistant0 tooling."""

    config: Dict[str, Any] | None = None
    if isinstance(candidate, str):
        try:
            config = json.loads(candidate)
        except json.JSONDecodeError:
            config = None
    elif isinstance(candidate, dict):
        config = dict(candidate)

    access_token = (config or {}).get("access_token") or os.getenv("ATLAS_AUTH0_ACCESS_TOKEN")
    refresh_token = (config or {}).get("refresh_token") or os.getenv("ATLAS_AUTH0_REFRESH_TOKEN")
    user = (config or {}).get("user") or {
        "sub": os.getenv("ATLAS_AUTH0_USER_SUB", "auth0|demo-user"),
        "email": os.getenv("ATLAS_AUTH0_USER_EMAIL", "demo@example.com"),
        "name": os.getenv("ATLAS_AUTH0_USER_NAME", "Demo User"),
    }

    if not access_token:
        access_token = "stub-access-token"

    credentials = {
        "access_token": access_token,
        "refresh_token": refresh_token or "stub-refresh-token",
        "user": user,
        "token_sets": (config or {}).get("token_sets")
        or [
            {
                "access_token": access_token,
                "scope": os.getenv("ATLAS_AUTH0_SCOPE", "openid profile email offline_access"),
            }
        ],
    }

    return credentials


__all__ = [
    "LangGraphAgentWrapper",
    "LangGraphEnvironment",
    "create_environment",
    "create_example_graph",
    "create_langgraph_agent",
]
