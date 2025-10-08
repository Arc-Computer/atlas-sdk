"""Deprecated wrapper around :mod:`atlas.connectors.factory`."""

from __future__ import annotations

import warnings

from atlas.connectors.factory import create_adapter, create_from_atlas_config

warnings.warn(
    "atlas.agent.factory is deprecated; import from atlas.connectors.factory",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["create_adapter", "create_from_atlas_config"]
