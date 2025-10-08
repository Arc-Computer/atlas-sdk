"""Deprecated shim for :mod:`atlas.runtime.orchestration.execution_context`."""

from __future__ import annotations

import warnings

from atlas.runtime.orchestration.execution_context import ExecutionContext  # noqa: F401

warnings.warn(
    "atlas.orchestration.execution_context is deprecated; import from atlas.runtime.orchestration.execution_context",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ExecutionContext"]
