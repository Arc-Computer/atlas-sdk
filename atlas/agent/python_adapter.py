"""Deprecated Python adapter shim importing from :mod:`atlas.connectors.python`."""

from __future__ import annotations

import warnings

from atlas.connectors.python import PythonAdapter

warnings.warn(
    "atlas.agent.python_adapter is deprecated; import from atlas.connectors.python",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["PythonAdapter"]
