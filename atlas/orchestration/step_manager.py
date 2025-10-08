"""Deprecated shim for :mod:`atlas.runtime.orchestration.step_manager`."""

from __future__ import annotations

import warnings

from atlas.runtime.orchestration.step_manager import IntermediateStepManager  # noqa: F401

warnings.warn(
    "atlas.orchestration.step_manager is deprecated; import from atlas.runtime.orchestration.step_manager",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["IntermediateStepManager"]
