"""Compatibility layer for legacy role imports.

The modern entry points live under :mod:`atlas.personas` and
:mod:`atlas.runtime.agent_loop`. Importing from ``atlas.roles`` emits a
deprecation warning but continues to work for now.
"""

from __future__ import annotations

import warnings

from atlas.personas.student import Student, StudentStepResult
from atlas.personas.teacher import Teacher
from atlas.connectors.langchain_bridge import BYOABridgeLLM, build_bridge
from atlas.runtime.agent_loop.tool_loop import ToolCallAgentGraph, ToolCallAgentGraphState

warnings.warn(
    "atlas.roles is deprecated; import from atlas.personas or atlas.runtime.agent_loop",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "BYOABridgeLLM",
    "Student",
    "StudentStepResult",
    "Teacher",
    "ToolCallAgentGraph",
    "ToolCallAgentGraphState",
    "build_bridge",
]
