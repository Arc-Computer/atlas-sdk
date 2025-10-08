"""Deprecated shim for :mod:`atlas.runtime.storage.database`."""

from __future__ import annotations

import warnings

from atlas.runtime.storage.database import *  # noqa: F401,F403

warnings.warn(
    "atlas.storage.database is deprecated; import from atlas.runtime.storage.database",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [name for name in globals() if not name.startswith("_")]
