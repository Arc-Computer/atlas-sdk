import json
from pathlib import Path

from atlas.evaluation.learning_report import LearningSummary, RewardSnapshot
from scripts.eval_learning import _slug_for_key, _write_outputs


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
