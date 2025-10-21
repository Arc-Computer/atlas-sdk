"""Adapter registry and session primitives for Bring-Your-Own-Agent integrations."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Dict, Mapping, MutableMapping, Protocol
from uuid import uuid4

from atlas.config.models import AdapterType
from atlas.config.models import AdapterUnion


class AdapterError(RuntimeError):
    """Raised when adapter execution fails."""


@dataclass(slots=True)
class SessionContext:
    """Container describing the runtime state for an adapter session."""

    task_id: str
    execution_mode: str
    tool_metadata: Mapping[str, Any] | None = None
    user_context: Mapping[str, Any] | None = None
    session_id: str | None = None
    extra: MutableMapping[str, Any] = field(default_factory=dict)


class AgentSession(Protocol):
    """Protocol implemented by adapter sessions."""

    session_id: str
    metadata: Mapping[str, Any]

    async def step(self, payload: str, metadata: Dict[str, Any] | None = None) -> Any:  # pragma: no cover - interface
        """Execute a single adapter step."""

    async def close(self, reason: str | None = None) -> Mapping[str, Any] | None:  # pragma: no cover - interface
        """Release resources associated with the session."""


@dataclass(slots=True)
class StatelessSession:
    """Fallback session that proxies calls to :meth:`AgentAdapter.ainvoke`."""

    adapter: "AgentAdapter"
    context: SessionContext
    _session_id: str = field(init=False)
    _metadata: dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        self._session_id = self.context.session_id or f"stateless-{uuid4()}"
        metadata: dict[str, Any] = {
            "mode": self.context.execution_mode,
            "task_id": self.context.task_id,
        }
        if self.context.user_context:
            metadata["user_context"] = dict(self.context.user_context)
        if self.context.tool_metadata:
            metadata["tool_metadata"] = dict(self.context.tool_metadata)
        metadata.update(self.context.extra)
        self._metadata = metadata

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._metadata

    async def step(self, payload: str, metadata: Dict[str, Any] | None = None) -> Any:
        merged = dict(metadata or {})
        merged.setdefault("adapter_session_id", self.session_id)
        return await self.adapter.ainvoke(payload, metadata=merged)

    async def close(self, reason: str | None = None) -> Mapping[str, Any] | None:  # pragma: no cover - trivial
        return None


class AgentAdapter:
    """Abstract adapter providing synchronous and asynchronous entrypoints."""

    supports_sessions: bool = False

    async def ainvoke(self, prompt: str, metadata: Dict[str, Any] | None = None) -> Any:
        raise NotImplementedError

    async def open_session(self, context: SessionContext) -> AgentSession:
        """Open a new session for adapters that support state."""

        return StatelessSession(self, context)

    def execute(self, prompt: str, metadata: Dict[str, Any] | None = None) -> Any:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(prompt, metadata))
        raise AdapterError("execute cannot be used inside a running event loop; use ainvoke instead")


class StatefulAgentAdapterMixin:
    """Mixin that provides convenience helpers for session-aware adapters."""

    supports_sessions: bool = True

    async def open_session(self, context: SessionContext) -> AgentSession:  # pragma: no cover - interface
        raise NotImplementedError

    @contextlib.asynccontextmanager
    async def session(self, context: SessionContext) -> AsyncIterator[AgentSession]:
        session = await self.open_session(context)
        try:
            yield session
        except Exception as exc:  # pragma: no cover - defensive cleanup
            with contextlib.suppress(Exception):
                await session.close(reason=str(exc))
            raise
        else:
            with contextlib.suppress(Exception):
                await session.close()


AdapterBuilder = Callable[[AdapterUnion], AgentAdapter]

_ADAPTER_BUILDERS: Dict[AdapterType, AdapterBuilder] = {}


def register_adapter(adapter_type: AdapterType, builder: AdapterBuilder) -> None:
    _ADAPTER_BUILDERS[adapter_type] = builder


def get_adapter_builder(adapter_type: AdapterType) -> AdapterBuilder:
    try:
        return _ADAPTER_BUILDERS[adapter_type]
    except KeyError as exc:  # pragma: no cover - defensive
        raise AdapterError(f"no adapter registered for type {adapter_type.value}") from exc


def build_adapter(config: AdapterUnion) -> AgentAdapter:
    builder = get_adapter_builder(config.type)
    return builder(config)


__all__ = [
    "AdapterError",
    "AgentAdapter",
    "AgentSession",
    "SessionContext",
    "StatefulAgentAdapterMixin",
    "StatelessSession",
    "register_adapter",
    "get_adapter_builder",
    "build_adapter",
]
