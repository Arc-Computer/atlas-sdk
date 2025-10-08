"""Compatibility shims for legacy adapter imports.

The new connector modules live under :mod:`atlas.connectors`. Importing from
``atlas.agent`` will continue to work but will emit a :class:`DeprecationWarning`.
"""

from __future__ import annotations

import warnings

from atlas.connectors.factory import create_adapter, create_from_atlas_config
from atlas.connectors.http import HTTPAdapter
from atlas.connectors.openai import OpenAIAdapter
from atlas.connectors.python import PythonAdapter
from atlas.connectors.registry import AdapterError, AgentAdapter


def _warn() -> None:
    warnings.warn(
        "atlas.agent is deprecated; import from atlas.connectors instead",
        DeprecationWarning,
        stacklevel=3,
    )


_warn()

__all__ = [
    "AdapterError",
    "AgentAdapter",
    "HTTPAdapter",
    "OpenAIAdapter",
    "PythonAdapter",
    "create_adapter",
    "create_from_atlas_config",
]
