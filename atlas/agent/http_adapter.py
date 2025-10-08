"""Deprecated HTTP adapter shim importing from :mod:`atlas.connectors.http`."""

from __future__ import annotations

import warnings

from atlas.connectors.http import HTTPAdapter

warnings.warn(
    "atlas.agent.http_adapter is deprecated; import from atlas.connectors.http",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["HTTPAdapter"]
