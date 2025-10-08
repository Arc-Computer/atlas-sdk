"""Deprecated shim for :mod:`atlas.runtime.orchestration.orchestrator`."""

from __future__ import annotations

import warnings

from atlas.runtime.orchestration.orchestrator import Orchestrator  # noqa: F401

warnings.warn(
    "atlas.orchestration.orchestrator is deprecated; import from atlas.runtime.orchestration.orchestrator",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["Orchestrator"]
