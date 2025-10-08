"""Deprecated shim for :mod:`atlas.runtime.storage`."""

from __future__ import annotations

import warnings

from atlas.runtime.storage.database import Database

warnings.warn(
    "atlas.storage is deprecated; import from atlas.runtime.storage",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["Database"]
