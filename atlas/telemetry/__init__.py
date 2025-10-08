"""Deprecated telemetry module."""

from __future__ import annotations

import warnings

from atlas.runtime.telemetry import ConsoleTelemetryStreamer

warnings.warn(
    "atlas.telemetry is deprecated; import from atlas.runtime.telemetry",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ConsoleTelemetryStreamer"]
