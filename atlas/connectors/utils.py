"""Shared helpers for connector implementations."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


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


__all__ = ["normalise_usage_payload"]
