"""Lightweight protocols used for autodiscovery of stateful agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Protocol, runtime_checkable


@dataclass(slots=True)
class DiscoveryEvent:
    """Normalized telemetry event emitted during discovery or runtime."""

    event_type: str
    payload: Dict[str, Any]


@dataclass(slots=True)
class DiscoveryContext:
    """Context provided to agents while the control loop executes."""

    task: str
    step_index: int
    observation: Any
    reward: Optional[float] = None
    done: bool = False


@runtime_checkable
class TelemetryEmitterProtocol(Protocol):
    """Telemetry callbacks exposed to the discovered environment/agent."""

    def emit(self, event_type: str, payload: Dict[str, Any] | None = None) -> None:
        ...

    def flush(self) -> None:
        ...


@runtime_checkable
class AtlasEnvironmentProtocol(Protocol):
    """Protocol the autodiscovery pipeline expects environments to implement."""

    def reset(self, task: str | None = None) -> Any:
        ...

    def step(self, action: Any) -> tuple[Any, float | None, bool, Dict[str, Any]]:
        ...

    def close(self) -> None:
        ...


@runtime_checkable
class AtlasAgentProtocol(Protocol):
    """Protocol the autodiscovery pipeline expects agents to implement."""

    def plan(
        self,
        task: str,
        observation: Any,
        *,
        emit_event: TelemetryEmitterProtocol | None = None,
    ) -> Any:
        ...

    def act(
        self,
        context: DiscoveryContext,
        *,
        emit_event: TelemetryEmitterProtocol | None = None,
    ) -> Any:
        ...

    def summarize(
        self,
        context: DiscoveryContext,
        *,
        history: Iterable[DiscoveryContext] | None = None,
        emit_event: TelemetryEmitterProtocol | None = None,
    ) -> str | None:
        ...
