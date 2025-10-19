"""Helpers for aggregating historical reward and learning signals."""

from __future__ import annotations

import os
from copy import deepcopy
from datetime import datetime
from typing import Any, Callable, Dict, Sequence

DEFAULT_HISTORY_LIMIT = 10
MAX_HISTORY_LIMIT = 200
MAX_NOTE_CHARS = 1024
_SENSITIVE_REWARD_KEYS = ("raw", "step_artifacts", "artifacts")
_HIGH_SCORE_THRESHOLD = 0.8
_LOW_SCORE_THRESHOLD = 0.4


def _normalise_timestamp(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        return value
    return None


def _resolve_limit(limit: int | None) -> int:
    env_limit = os.getenv("ATLAS_LEARNING_HISTORY_LIMIT")
    if env_limit:
        try:
            parsed = int(env_limit)
        except ValueError:
            parsed = None
        if parsed and parsed > 0:
            return min(parsed, MAX_HISTORY_LIMIT)
    if limit is None or limit <= 0:
        return DEFAULT_HISTORY_LIMIT
    return min(limit, MAX_HISTORY_LIMIT)


def _truncate(value: Any, max_chars: int) -> Any:
    if not isinstance(value, str):
        return value
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "..."


def _sanitise_reward(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    reward = deepcopy(payload)
    for key in _SENSITIVE_REWARD_KEYS:
        reward.pop(key, None)
    return reward


def _extract_score(payload: Any) -> float | None:
    if not isinstance(payload, dict):
        return None
    score = payload.get("score")
    if isinstance(score, (int, float)):
        return float(score)
    return None


def _compute_streak(scores: Sequence[float], predicate: Callable[[float], bool]) -> int:
    streak = 0
    for score in reversed(scores):
        if predicate(score):
            streak += 1
        else:
            break
    return streak


def _build_entry(record: Dict[str, Any], *, max_chars: int) -> tuple[dict[str, Any], float | None]:
    reward_payload = _sanitise_reward(record.get("reward"))
    score = _extract_score(reward_payload)
    entry = {
        "reward": reward_payload,
        "student_learning": _truncate(record.get("student_learning"), max_chars),
        "teacher_learning": _truncate(record.get("teacher_learning"), max_chars),
        "created_at": _normalise_timestamp(record.get("created_at")),
        "completed_at": _normalise_timestamp(record.get("completed_at")),
    }
    return entry, score


def aggregate_learning_history(
    records: Sequence[Dict[str, Any]] | None,
    *,
    limit: int | None = None,
    max_note_chars: int = MAX_NOTE_CHARS,
) -> Dict[str, Any]:
    """
    Aggregate prior reward and learning entries into a compact, probe-friendly payload.

    The returned dictionary includes:
    - ``entries``: up to ``limit`` most recent records with sanitised reward data and truncated notes.
    - ``scores`` / ``average_score``: per-entry scores and their average across the limited window.
    - ``overall_average_score``: average score across the full record set (when available).
    - ``recent_high_score_streak`` / ``recent_low_score_streak``: consecutive high/low scores within the window.
    - ``count``: number of entries included after trimming.
    - ``total_count``: total number of records supplied before trimming.
    - ``limit``: the effective limit used (respecting config/environment constraints).

    Notes:
    - ``limit`` defaults to :data:`DEFAULT_HISTORY_LIMIT` and respects the
      ``ATLAS_LEARNING_HISTORY_LIMIT`` environment override as well as an upper bound of
      :data:`MAX_HISTORY_LIMIT`.
    - Reward payloads are copied with heavyweight keys removed (``raw``, ``step_artifacts``, ``artifacts``).
    - ``student_learning`` / ``teacher_learning`` notes are truncated to ``max_note_chars`` characters.
    """

    total_count = len(records) if records else 0
    if not records:
        return {"entries": [], "count": 0, "total_count": 0}

    resolved_limit = _resolve_limit(limit)
    limited_records = list(records)[-resolved_limit:]

    entries: list[dict[str, Any]] = []
    recent_scores: list[float] = []
    all_scores: list[float] = []

    for record in records:
        score = _extract_score(record.get("reward"))
        if score is not None:
            all_scores.append(score)

    for record in limited_records:
        entry, score = _build_entry(record, max_chars=max_note_chars)
        entries.append(entry)
        if score is not None:
            recent_scores.append(score)

    aggregated: dict[str, Any] = {
        "entries": entries,
        "count": len(entries),
        "total_count": total_count,
    }

    if recent_scores:
        aggregated["scores"] = recent_scores
        aggregated["average_score"] = sum(recent_scores) / len(recent_scores)
        aggregated["recent_high_score_streak"] = _compute_streak(
            recent_scores, lambda value: value >= _HIGH_SCORE_THRESHOLD
        )
        aggregated["recent_low_score_streak"] = _compute_streak(
            recent_scores, lambda value: value <= _LOW_SCORE_THRESHOLD
        )
    else:
        aggregated["recent_high_score_streak"] = 0
        aggregated["recent_low_score_streak"] = 0

    if all_scores:
        aggregated["overall_average_score"] = sum(all_scores) / len(all_scores)

    aggregated["limit"] = resolved_limit
    return aggregated
