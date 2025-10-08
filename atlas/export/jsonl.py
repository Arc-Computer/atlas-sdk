"""Deprecated JSONL exporter shim."""

from __future__ import annotations

import warnings

from atlas.cli.jsonl_writer import *  # noqa: F401,F403

warnings.warn(
    "atlas.export.jsonl is deprecated; import from atlas.cli.jsonl_writer",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [name for name in globals() if not name.startswith("_")]
