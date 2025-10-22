import json
import pytest
from pathlib import Path

from atlas.evaluation.learning_report import (
    LearningModelBreakdown,
    LearningSummary,
    RewardSnapshot,
    WindowSpec,
)
from scripts.eval_learning import _compute_comparisons, _slug_for_key, _write_outputs


class DummySummary(LearningSummary):
    pass


def _make_summary(key: str) -> LearningSummary:
    return LearningSummary(
        learning_key=key,
        session_count=1,
        reward=RewardSnapshot(
            recent_mean=0.9,
            recent_count=1,
            baseline_mean=0.8,
            baseline_count=10,
            delta=0.1,
            latest_score=0.95,
            recent_window=WindowSpec(label="recent", size=1),
            baseline_window=WindowSpec(label="baseline", size=10),
        ),
    )


def test_slug_for_key_sanitises_input():
    assert _slug_for_key("abc123") == "abc123"
    slug = _slug_for_key("!!invalid!!")
    assert slug == "invalid"


def test_write_outputs_creates_files(tmp_path: Path):
    summary = _make_summary("abc123")
    manifest = _write_outputs([summary], tmp_path, write_markdown=True)
    assert "abc123" in manifest
    json_path = Path(manifest["abc123"]["json"])
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["learning_key"] == "abc123"
    markdown_path = Path(manifest["abc123"]["markdown"])
    assert markdown_path.exists()
    assert "Learning Evaluation" in markdown_path.read_text(encoding="utf-8")


def test_write_outputs_with_comparisons(tmp_path: Path):
    summary = _make_summary("xyz789")
    summary.model_breakdown = [
        LearningModelBreakdown(
            role="student",
            model_id="model-a",
            session_count=1,
            reward_count=1,
            reward_mean=0.9,
            latest_score=0.9,
        )
    ]
    comparisons = {
        "xyz789": {
            "recent_mean_delta": 0.05,
            "latest_score_delta": -0.02,
            "session_count_delta": 1,
            "model_deltas": {
                "student:model-a": {
                    "session_count_delta": 1,
                    "reward_mean_delta": 0.05,
                }
            },
        }
    }
    aggregate = {"recent_mean_best": [{"learning_key": "xyz789", "delta": 0.05}]}
    manifest = _write_outputs(
        [summary],
        tmp_path,
        write_markdown=True,
        comparisons=comparisons,
        aggregate=aggregate,
    )
    json_path = Path(manifest["xyz789"]["json"])
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert "comparison" in payload
    markdown_path = Path(manifest["xyz789"]["markdown"])
    text = markdown_path.read_text(encoding="utf-8")
    assert "Comparison vs previous run" in text
    assert "recent mean delta" in text.lower()


def test_compute_comparisons_returns_deltas():
    summary = _make_summary("delta-key")
    summary.session_count = 3
    summary.reward.recent_mean = 1.0
    summary.reward.latest_score = 1.1
    summary.model_breakdown = [
        LearningModelBreakdown(
            role="student",
            model_id="model-z",
            session_count=3,
            reward_count=3,
            reward_mean=1.0,
            latest_score=1.1,
        )
    ]
    previous_payloads = {
        "delta-key": {
            "session_count": 2,
            "reward": {"recent_mean": 0.5, "latest_score": 0.7},
            "model_breakdown": [
                {
                    "role": "student",
                    "model_id": "model-z",
                    "session_count": 2,
                    "reward_mean": 0.4,
                }
            ],
        }
    }
    comparisons, aggregate = _compute_comparisons([summary], previous_payloads)
    entry = comparisons["delta-key"]
    assert pytest.approx(entry["recent_mean_delta"], rel=1e-4) == 0.5
    assert pytest.approx(entry["latest_score_delta"], rel=1e-4) == 0.4
    assert entry["session_count_delta"] == 1
    model_delta = entry["model_deltas"]["student:model-z"]
    assert model_delta["session_count_delta"] == 1
    assert pytest.approx(model_delta["reward_mean_delta"], rel=1e-4) == 0.6
    assert aggregate["recent_mean_best"][0]["learning_key"] == "delta-key"
