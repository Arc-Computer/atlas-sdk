"""Deprecated shim for :mod:`atlas.personas.student`."""

from __future__ import annotations

import warnings

from atlas.personas.student import Student, StudentStepResult

warnings.warn(
    "atlas.roles.student is deprecated; import from atlas.personas.student",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["Student", "StudentStepResult"]
