"""Helpers for injecting learning pamphlets into runtime prompts."""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any, Dict, List, Tuple

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


def extract_few_shot_examples(
    metadata: Dict[str, Any] | None,
    role: str,
    *,
    max_tokens: int = 500,
    redaction_patterns: List[str] | None = None,
    chars_per_token: float = 3.5,
    max_entries: int = 10,
    max_examples_per_block: int = 2,
) -> str | None:
    """Extract and format few-shot examples from learning_usage.

    Args:
        metadata: Playbook metadata dictionary (not currently used for example extraction)
        role: Either "student" or "teacher" to select which examples to extract
        max_tokens: Approximate token budget for few-shot examples
        redaction_patterns: List of regex patterns for redacting sensitive data
        chars_per_token: Conservative multiplier for token-to-char conversion (default 3.5)
        max_entries: Maximum number of learning entries to process (default 10)
        max_examples_per_block: Maximum examples per cue/adoption block (default 2)

    Returns:
        Formatted few-shot examples string, or None if no examples available

    Example output:
        >>> Few-Shot Examples >>>
        Entry abc123:
          Cue examples:
            1. investigating latency issues
          Action examples:
            1. metrics.query -> success
        >>> End Few-Shot Examples >>>
    """

    if not isinstance(metadata, dict):
        return None
    if max_tokens <= 0:
        return None

    try:
        context = ExecutionContext.get()
        learning_usage = context.metadata.get("learning_usage", {})
    except Exception:
        return None

    if not isinstance(learning_usage, dict):
        return None

    role_usage = learning_usage.get("roles", {}).get(role, {})
    if not isinstance(role_usage, dict):
        return None

    examples_blocks: List[str] = []
    char_budget = int(max_tokens * chars_per_token)  # Conservative estimate with safety margin
    char_used = 0
    entries_processed = 0

    for entry_id, entry_data in role_usage.items():
        if entries_processed >= max_entries:
            break

        if not isinstance(entry_data, dict):
            continue

        cue_examples = entry_data.get("cue_examples", []) or []
        adoption_examples = entry_data.get("adoption_examples", []) or []
        if not cue_examples and not adoption_examples:
            continue

        block_lines: List[str] = [f"Entry {entry_id}:"]

        if cue_examples:
            block_lines.append("  Cue examples:")
            for idx, example in enumerate(cue_examples[:max_examples_per_block], start=1):
                redacted = _redact_sensitive_data(str(example), redaction_patterns)
                block_lines.append(f"    {idx}. {redacted}")

        if adoption_examples:
            block_lines.append("  Action examples:")
            for idx, example in enumerate(adoption_examples[:max_examples_per_block], start=1):
                if isinstance(example, dict):
                    tool = example.get("tool_name") or example.get("runtime_handle") or "unknown"
                    status = example.get("status") or ("success" if example.get("success") else "unknown")
                    block_lines.append(f"    {idx}. {tool} -> {status}")
                else:
                    redacted = _redact_sensitive_data(str(example), redaction_patterns)
                    block_lines.append(f"    {idx}. {redacted}")

        block_text = "\n".join(block_lines).strip()
        block_chars = len(block_text)
        if char_used + block_chars > char_budget:
            break

        examples_blocks.append(block_text)
        char_used += block_chars
        entries_processed += 1

    if not examples_blocks:
        return None

    # Use list + join pattern for better performance
    parts = [
        ">>> Few-Shot Examples >>>",
        "\n\n".join(examples_blocks),
        ">>> End Few-Shot Examples >>>",
    ]
    return "\n".join(parts)


def _redact_sensitive_data(text: str, patterns: List[str] | None) -> str:
    """Redact sensitive data using regex patterns."""
    if not patterns:
        return text
    redacted = text
    for pattern in patterns:
        try:
            redacted = re.sub(pattern, "[REDACTED]", redacted, flags=re.IGNORECASE)
        except re.error:
            continue
    return redacted
