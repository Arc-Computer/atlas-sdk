"""Deprecated export utilities."""

from __future__ import annotations

import warnings

from atlas.cli.jsonl_writer import (
    ExportRequest,
    ExportSummary,
    ExportStats,
    export_sessions,
    export_sessions_async,
    export_sessions_sync,
    export_sessions_to_jsonl,
)
from atlas.cli.export import main

warnings.warn(
    "atlas.export is deprecated; import from atlas.cli",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "ExportRequest",
    "ExportSummary",
    "ExportStats",
    "export_sessions",
    "export_sessions_async",
    "export_sessions_sync",
    "export_sessions_to_jsonl",
    "main",
]
