"""Helpers for injecting learning pamphlets into runtime prompts."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, Tuple

from atlas.learning.usage import get_tracker
from atlas.runtime.orchestration.execution_context import ExecutionContext

logger = logging.getLogger(__name__)


def resolve_playbook(
    role: str,
    *,
    apply: bool,
    limit: int = 1000,
) -> Tuple[str | None, str | None, Dict[str, Any] | None]:
    """Fetch and normalise the cached pamphlet for the requested role.

    Parameters
    ----------
    role:
        Either ``"student"`` or ``"teacher"`` to address the corresponding
        pamphlet stored inside ``ExecutionContext.metadata['learning_state']``.
    apply:
        Controls whether pamphlets should be surfaced. When ``False`` the
        function always returns ``(None, None, None)``.
    limit:
        Maximum number of characters to include in prompts. When the pamphlet
        exceeds this budget the helper trims the payload and logs the event.

    Returns
    -------
    tuple[str | None, str | None, dict[str, Any] | None]
        ``(pamphlet, digest, metadata)`` where ``digest`` is a SHA-256 hex
        string of the trimmed pamphlet and ``metadata`` mirrors the optional
        ``learning_state['metadata']`` dictionary.
    """

    if not apply:
        return None, None, None

    try:
        context = ExecutionContext.get()
    except Exception:  # pragma: no cover - defensive guard when unset
        return None, None, None

    state = context.metadata.get("learning_state")
    if not isinstance(state, dict):
        return None, None, None

    key = f"{role}_learning"
    raw_value = state.get(key)
    metadata = state.get("metadata") if isinstance(state.get("metadata"), dict) else None

    if isinstance(metadata, dict):
        try:
            tracker = get_tracker()
            entries = metadata.get("playbook_entries")
            tracker.register_entries(role, entries or [])
        except Exception:  # pragma: no cover - instrumentation must not fail core flow
            logger.debug("Unable to register playbook entries for role %s", role, exc_info=True)

    cache = context.metadata.setdefault("_learning_playbooks", {})
    cached = cache.get(role)
    if cached and cached.get("raw") == raw_value:
        return cached.get("text"), cached.get("digest"), metadata

    if not isinstance(raw_value, str):
        cache[role] = {"raw": raw_value, "text": None, "digest": None}
        return None, None, metadata

    trimmed = raw_value.strip()
    if not trimmed:
        cache[role] = {"raw": raw_value, "text": None, "digest": None}
        return None, None, metadata

    if len(trimmed) > limit:
        logger.info(
            "%s playbook trimmed from %s to %s characters", role.title(), len(trimmed), limit
        )
        trimmed = trimmed[: limit - 3].rstrip()
        trimmed = f"{trimmed}..."

    digest = hashlib.sha256(trimmed.encode("utf-8")).hexdigest()
    cache[role] = {"raw": raw_value, "text": trimmed, "digest": digest}

    # Record that learning was applied for adoption tracking
    # Metadata must be written to session_metadata which gets persisted to the database
    applied_key = f"applied_{role}_learning"
    entry_count = len(metadata.get("playbook_entries", [])) if isinstance(metadata, dict) else 0
    session_meta = context.metadata.setdefault("session_metadata", {})
    session_meta[applied_key] = {
        "digest": digest,
        "char_count": len(trimmed),
        "entry_count": entry_count,
    }

    return trimmed, digest, metadata
