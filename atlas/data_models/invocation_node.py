"""Deprecated shim for :mod:`atlas.runtime.models.invocation_node`."""

from __future__ import annotations

import warnings

from atlas.runtime.models.invocation_node import *  # noqa: F401,F403

warnings.warn(
    "atlas.data_models.invocation_node is deprecated; import from atlas.runtime.models.invocation_node",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [name for name in globals() if not name.startswith("_")]
