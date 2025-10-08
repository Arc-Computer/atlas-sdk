"""Deprecated shim for :mod:`atlas.connectors.langchain_bridge`."""

from __future__ import annotations

import warnings

from atlas.connectors.langchain_bridge import BYOABridgeLLM, build_bridge

warnings.warn(
    "atlas.roles.student_bridge is deprecated; import from atlas.connectors.langchain_bridge",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["BYOABridgeLLM", "build_bridge"]
