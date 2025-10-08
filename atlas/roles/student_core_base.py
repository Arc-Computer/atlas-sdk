"""Deprecated shim for :mod:`atlas.runtime.agent_loop.base_agent`."""

from __future__ import annotations

import warnings

from atlas.runtime.agent_loop.base_agent import (
    AGENT_CALL_LOG_MESSAGE,
    AGENT_LOG_PREFIX,
    AgentDecision,
    BaseAgent,
)

warnings.warn(
    "atlas.roles.student_core_base is deprecated; import from atlas.runtime.agent_loop.base_agent",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "AGENT_CALL_LOG_MESSAGE",
    "AGENT_LOG_PREFIX",
    "AgentDecision",
    "BaseAgent",
]
