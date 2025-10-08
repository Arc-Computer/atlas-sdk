"""Deprecated shim for :mod:`atlas.runtime.telemetry.console`."""

from __future__ import annotations

import warnings

from atlas.runtime.telemetry.console import *  # noqa: F401,F403

warnings.warn(
    "atlas.telemetry.console is deprecated; import from atlas.runtime.telemetry.console",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [name for name in globals() if not name.startswith("_")]
