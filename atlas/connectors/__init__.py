"""Connector adapters for integrating Bring-Your-Own-Agent implementations."""

from .factory import create_adapter, create_from_atlas_config
from .registry import (
    AdapterError,
    AgentAdapter,
    build_adapter,
    register_adapter,
)

# Import adapters to trigger registration
from . import http, openai, python  # noqa: F401

__all__ = [
    "AdapterError",
    "AgentAdapter",
    "build_adapter",
    "create_adapter",
    "create_from_atlas_config",
    "register_adapter",
]
