"""Deprecated shim for :mod:`atlas.runtime.agent_loop.tool_loop`."""

from __future__ import annotations

import warnings

from atlas.runtime.agent_loop.tool_loop import ToolCallAgentGraph, ToolCallAgentGraphState

warnings.warn(
    "atlas.roles.student_core is deprecated; import from atlas.runtime.agent_loop.tool_loop",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ToolCallAgentGraph", "ToolCallAgentGraphState"]
