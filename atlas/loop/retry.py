"""Deprecated shim importing from :mod:`atlas.runtime.agent_loop.retries`."""

from __future__ import annotations

import warnings

from atlas.runtime.agent_loop.retries import *  # noqa: F401,F403

warnings.warn(
    "atlas.loop.retry is deprecated; import from atlas.runtime.agent_loop.retries",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [name for name in globals() if name.isupper() or name.endswith("Retry")]
