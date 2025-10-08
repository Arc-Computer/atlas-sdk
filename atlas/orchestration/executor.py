"""Deprecated shim for :mod:`atlas.runtime.orchestration.executor`."""

from __future__ import annotations

import warnings

from atlas.runtime.orchestration.executor import *  # noqa: F401,F403

warnings.warn(
    "atlas.orchestration.executor is deprecated; import from atlas.runtime.orchestration.executor",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [name for name in globals() if not name.startswith("_")]
