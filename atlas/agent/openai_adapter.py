"""Deprecated OpenAI adapter shim importing from :mod:`atlas.connectors.openai`."""

from __future__ import annotations

import warnings

from atlas.connectors.openai import OpenAIAdapter

warnings.warn(
    "atlas.agent.openai_adapter is deprecated; import from atlas.connectors.openai",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["OpenAIAdapter"]
