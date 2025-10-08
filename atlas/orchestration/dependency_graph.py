"""Deprecated shim for :mod:`atlas.runtime.orchestration.dependency_graph`."""

from __future__ import annotations

import warnings

from atlas.runtime.orchestration.dependency_graph import DependencyGraph  # noqa: F401

warnings.warn(
    "atlas.orchestration.dependency_graph is deprecated; import from atlas.runtime.orchestration.dependency_graph",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["DependencyGraph"]
