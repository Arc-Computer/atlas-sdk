"""Deprecated CLI shim."""

from __future__ import annotations

import warnings

from atlas.cli.export import main  # noqa: F401

warnings.warn(
    "atlas.export.cli is deprecated; import from atlas.cli.export",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["main"]
