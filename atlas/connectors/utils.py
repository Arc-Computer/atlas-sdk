"""Shared helpers for connector implementations."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Sequence


class AdapterResponse(str):
    """String-compatible adapter payload that exposes optional metadata."""

    def __new__(
        cls,
        content: str,
        *,
        tool_calls: Sequence[Dict[str, Any]] | None = None,
        usage: Optional[Dict[str, Any]] = None,
    ) -> "AdapterResponse":
        obj = super().__new__(cls, content or "")
        obj.tool_calls = tool_calls
        obj.usage = usage
        return obj

    tool_calls: Sequence[Dict[str, Any]] | None
    usage: Optional[Dict[str, Any]]

    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


def normalise_usage_payload(usage: Any) -> Optional[Dict[str, Any]]:
    """Convert LiteLLM usage payloads into plain dictionaries."""
    if usage is None:
        return None
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        try:
            payload = usage.model_dump()
            if isinstance(payload, dict):
                return payload
        except Exception:
            return None
    if hasattr(usage, "dict"):
        try:
            payload = usage.dict()
            if isinstance(payload, dict):
                return payload
        except Exception:
            return None
    if isinstance(usage, str):
        try:
            payload = json.loads(usage)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            return None
    return None


__all__ = ["AdapterResponse", "normalise_usage_payload"]
