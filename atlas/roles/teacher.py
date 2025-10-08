"""Deprecated shim for :mod:`atlas.personas.teacher`."""

from __future__ import annotations

import warnings

from atlas.personas.teacher import Teacher

warnings.warn(
    "atlas.roles.teacher is deprecated; import from atlas.personas.teacher",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["Teacher"]
