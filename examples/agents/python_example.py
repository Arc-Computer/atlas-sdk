"""Minimal callable used by the Python adapter example."""

from __future__ import annotations

from typing import Any, Dict


def run_agent(prompt: str, metadata: Dict[str, Any] | None = None) -> str:
    """Return a canned response that echoes the prompt with context."""
    metadata = metadata or {}
    tags = ", ".join(f"{key}={value}" for key, value in metadata.items()) or "no metadata"
    return (
        "Python agent received:\n"
        f"- prompt: {prompt}\n"
        f"- metadata: {tags}\n"
        "Pretend this section contains real business logic."
    )

