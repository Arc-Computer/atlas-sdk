"""Deprecated shim for :mod:`atlas.runtime.agent_loop.dual_node`."""

from __future__ import annotations

import warnings

from atlas.runtime.agent_loop.dual_node import DualNodeAgent

warnings.warn(
    "atlas.roles.dual_node_base is deprecated; import from atlas.runtime.agent_loop.dual_node",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["DualNodeAgent"]
