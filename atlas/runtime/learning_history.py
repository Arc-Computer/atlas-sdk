"""Helpers for aggregating historical reward and learning signals."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Sequence


def _normalise_timestamp(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        return value
    return None


def aggregate_learning_history(
    records: Sequence[Dict[str, Any]] | None,
    learning_state: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Aggregate prior reward and learning entries plus current registry snapshot."""

    entries: list[dict[str, Any]] = []
    scores: list[float] = []

    if records:
        for record in records:
            reward_payload = record.get("reward")
            if isinstance(reward_payload, dict):
                raw_score = reward_payload.get("score")
                if isinstance(raw_score, (int, float)):
                    scores.append(float(raw_score))
            entries.append(
                {
                    "reward": reward_payload,
                    "student_learning": record.get("student_learning"),
                    "teacher_learning": record.get("teacher_learning"),
                    "created_at": _normalise_timestamp(record.get("created_at")),
                    "completed_at": _normalise_timestamp(record.get("completed_at")),
                }
            )

    aggregated: dict[str, Any] = {"entries": entries, "count": len(entries)}
    if scores:
        aggregated["scores"] = scores
        aggregated["average_score"] = sum(scores) / len(scores)
    if isinstance(learning_state, dict):
        student_doc = learning_state.get("student_learning")
        teacher_doc = learning_state.get("teacher_learning")
        if isinstance(student_doc, str) and student_doc.strip():
            aggregated["current_student_learning"] = student_doc
        if isinstance(teacher_doc, str) and teacher_doc.strip():
            aggregated["current_teacher_learning"] = teacher_doc
        metadata = learning_state.get("metadata")
        if isinstance(metadata, dict) and metadata:
            aggregated["current_metadata"] = metadata
        timestamp = _normalise_timestamp(learning_state.get("updated_at"))
        if timestamp:
            aggregated["current_updated_at"] = timestamp
    return aggregated
