"""Deprecated shim for :mod:`atlas.runtime.telemetry.langchain_callback`."""

from __future__ import annotations

import warnings

from atlas.runtime.telemetry.langchain_callback import *  # noqa: F401,F403

warnings.warn(
    "atlas.telemetry.langchain_callback is deprecated; import from atlas.runtime.telemetry.langchain_callback",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [name for name in globals() if not name.startswith("_")]
