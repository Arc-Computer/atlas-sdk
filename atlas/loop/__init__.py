"""Deprecated retry helpers. Use :mod:`atlas.runtime.agent_loop.retries`."""

from __future__ import annotations

import warnings

warnings.warn(
    "atlas.loop is deprecated; import from atlas.runtime.agent_loop",
    DeprecationWarning,
    stacklevel=2,
)
