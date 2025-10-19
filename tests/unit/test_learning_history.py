from __future__ import annotations

from datetime import datetime

import pytest

from atlas.runtime.learning_history import (
    DEFAULT_HISTORY_LIMIT,
    MAX_NOTE_CHARS,
    aggregate_learning_history,
)


def _make_record(score: float, student_note: str = "note", teacher_note: str = "coach") -> dict:
    return {
        "reward": {
            "score": score,
            "raw": {"details": "large blob"},
            "judges": [{"name": "process", "score": score}],
        },
        "student_learning": student_note,
        "teacher_learning": teacher_note,
        "created_at": datetime(2024, 1, 1, 12, 0, 0),
        "completed_at": datetime(2024, 1, 1, 12, 5, 0),
    }


def test_aggregate_learning_history_trims_and_sanitises() -> None:
    records = [_make_record(0.1 * idx, student_note="s" * (MAX_NOTE_CHARS + 100)) for idx in range(1, 6)]

    result = aggregate_learning_history(records, limit=3)

    assert result["count"] == 3
    assert result["total_count"] == 5
    assert result["entries"][0]["reward"]["score"] == pytest.approx(0.3)
    assert "raw" not in result["entries"][0]["reward"]
    note = result["entries"][0]["student_learning"]
    assert isinstance(note, str)
    assert len(note) <= MAX_NOTE_CHARS + 3  # truncated with ellipsis


def test_aggregate_learning_history_scores_and_streaks() -> None:
    records = [
        _make_record(1.0),
        _make_record(0.9),
        _make_record(0.3),
    ]

    result = aggregate_learning_history(records, limit=DEFAULT_HISTORY_LIMIT)

    assert result["scores"] == [1.0, 0.9, 0.3]
    assert result["average_score"] == pytest.approx((1.0 + 0.9 + 0.3) / 3)
    assert result["overall_average_score"] == pytest.approx((1.0 + 0.9 + 0.3) / 3)
    assert result["recent_high_score_streak"] == 0
    assert result["recent_low_score_streak"] == 1


def test_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    records = [_make_record(0.5) for _ in range(5)]
    monkeypatch.setenv("ATLAS_LEARNING_HISTORY_LIMIT", "2")

    result = aggregate_learning_history(records, limit=DEFAULT_HISTORY_LIMIT)

    assert result["count"] == 2
    assert result["limit"] == 2
