from __future__ import annotations

import hashlib
import json
from typing import Any


def _normalise(value: Any, *, depth: int = 0) -> Any:
    if depth > 6:
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        normalised: dict[str, Any] = {}
        for key, item in value.items():
            normalised[str(key)] = _normalise(item, depth=depth + 1)
        return normalised
    if isinstance(value, (list, tuple, set)):
        return [_normalise(item, depth=depth + 1) for item in value]
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
        except Exception:  # pragma: no cover - defensive fallback
            return str(value)
        return _normalise(dumped, depth=depth + 1)
    if hasattr(value, "to_dict"):
        try:
            dumped = value.to_dict()
        except Exception:  # pragma: no cover - defensive fallback
            return str(value)
        return _normalise(dumped, depth=depth + 1)
    if hasattr(value, "__dict__"):
        return _normalise(vars(value), depth=depth + 1)
    return str(value)


def normalise_json(value: Any) -> Any:
    """Normalise arbitrary structures into JSON-serialisable data."""

    return _normalise(value)


def json_digest(value: Any) -> str:
    """Compute a deterministic SHA-256 digest for arbitrary JSON-like payloads."""

    normalised = normalise_json(value)
    encoded = json.dumps(normalised, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
