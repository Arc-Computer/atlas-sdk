"""Deprecated shim for :mod:`atlas.runtime.models.intermediate_step`."""

from __future__ import annotations

import warnings

from atlas.runtime.models.intermediate_step import *  # noqa: F401,F403

warnings.warn(
    "atlas.data_models.intermediate_step is deprecated; import from atlas.runtime.models.intermediate_step",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [name for name in globals() if not name.startswith("_")]
