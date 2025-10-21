"""Adapter registry for Bring-Your-Own-Agent integrations."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field

from atlas.config.models import AdapterType
from atlas.config.models import AdapterUnion


class AdapterError(RuntimeError):
    """Raised when adapter execution fails."""


AdapterControlLoop = Literal["atlas", "self"]
AdapterEventEmitter = Callable[[Dict[str, Any]], Awaitable[None] | None]


class AdapterCapabilities(BaseModel):
    """Negotiated adapter capabilities for the current session."""

    model_config = ConfigDict(extra="allow")

    control_loop: AdapterControlLoop = Field(default="atlas")
    supports_stepwise: bool = Field(default=True)
    telemetry_stream: bool = Field(default=False)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class AgentAdapter:
    """Abstract adapter providing synchronous and asynchronous entrypoints."""

    async def ainvoke(self, prompt: str, metadata: Dict[str, Any] | None = None) -> str:
        raise NotImplementedError

    async def aopen_session(
        self,
        *,
        task: str,
        metadata: Dict[str, Any] | None = None,
        emit_event: AdapterEventEmitter | None = None,
    ) -> AdapterCapabilities:
        """Negotiate session capabilities. Stateless adapters return defaults."""

        _ = task, metadata, emit_event  # intentionally unused defaults
        return AdapterCapabilities()

    async def aplan(self, task: str, metadata: Dict[str, Any] | None = None) -> Any:
        raise AdapterError("adapter does not implement aplan() for self-managed control loop")

    async def aexecute(
        self,
        task: str,
        plan: Dict[str, Any],
        step: Dict[str, Any],
        metadata: Dict[str, Any] | None = None,
    ) -> Any:
        raise AdapterError("adapter does not implement aexecute() for self-managed control loop")

    async def asynthesize(
        self,
        task: str,
        plan: Dict[str, Any],
        step_results: List[Dict[str, Any]],
        metadata: Dict[str, Any] | None = None,
    ) -> str:
        raise AdapterError("adapter does not implement asynthesize() for self-managed control loop")

    def execute(self, prompt: str, metadata: Dict[str, Any] | None = None) -> str:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(prompt, metadata))
        raise AdapterError("execute cannot be used inside a running event loop; use ainvoke instead")


AdapterBuilder = Callable[[AdapterUnion], AgentAdapter]

_ADAPTER_BUILDERS: Dict[AdapterType, AdapterBuilder] = {}


def register_adapter(adapter_type: AdapterType, builder: AdapterBuilder) -> None:
    _ADAPTER_BUILDERS[adapter_type] = builder


def get_adapter_builder(adapter_type: AdapterType) -> AdapterBuilder:
    try:
        return _ADAPTER_BUILDERS[adapter_type]
    except KeyError as exc:
        raise AdapterError(f"no adapter registered for type {adapter_type.value}") from exc


def build_adapter(config: AdapterUnion) -> AgentAdapter:
    builder = get_adapter_builder(config.type)
    return builder(config)


__all__ = [
    "AdapterCapabilities",
    "AdapterControlLoop",
    "AdapterError",
    "AgentAdapter",
    "AdapterEventEmitter",
    "register_adapter",
    "get_adapter_builder",
    "build_adapter",
]
