"""Deprecated registry shim importing from :mod:`atlas.connectors.registry`."""

from __future__ import annotations

import warnings

from atlas.connectors.registry import (  # noqa: F401
    AdapterError,
    AgentAdapter,
    build_adapter,
    get_adapter_builder,
    register_adapter,
)

warnings.warn(
    "atlas.agent.registry is deprecated; import from atlas.connectors.registry",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "AdapterError",
    "AgentAdapter",
    "build_adapter",
    "get_adapter_builder",
    "register_adapter",
]
