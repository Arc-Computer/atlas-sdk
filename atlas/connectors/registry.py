"""Adapter registry for Bring-Your-Own-Agent integrations."""

from __future__ import annotations

import asyncio
from typing import Any
from typing import Callable
from typing import Dict

from atlas.config.models import AdapterType
from atlas.config.models import AdapterUnion

class AdapterError(RuntimeError):
    """Raised when adapter execution fails."""


class AgentAdapter:
    """Abstract adapter providing synchronous and asynchronous entrypoints."""

    async def ainvoke(self, prompt: str, metadata: Dict[str, Any] | None = None) -> str:
        raise NotImplementedError

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
    "AdapterError",
    "AgentAdapter",
    "register_adapter",
    "get_adapter_builder",
    "build_adapter",
]
