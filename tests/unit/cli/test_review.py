from datetime import datetime, timezone

from atlas.cli.review import ReviewSession, format_review_groups, format_session_summary


def _build_session(**overrides) -> ReviewSession:
    defaults = {
        "identifier": 42,
        "task": "Investigate drift alert",
        "status": "succeeded",
        "review_status": "pending",
        "review_notes": None,
        "created_at": datetime(2025, 2, 1, 12, 0, tzinfo=timezone.utc),
        "drift_alert": True,
        "drift": {"score_delta": 0.4, "uncertainty_delta": -0.2, "reason": "score_z", "drift_alert": True},
        "reward_stats": {"score": 0.9, "score_stddev": 0.05, "sample_count": 3},
        "metadata": {},
        "reward_audit": [],
    }
    defaults.update(overrides)
    return ReviewSession(**defaults)


def test_format_review_groups_includes_drift_metrics():
    session = _build_session()
    lines = format_review_groups({"pending": [session]})

    summary_line = next(line for line in lines if "scoreΔ" in line)
    assert "ALERT" in summary_line
    assert "scoreΔ=+0.40" in summary_line
    assert "uncΔ=-0.20" in summary_line
    assert "reward=0.90±0.05 (n=3)" in summary_line


def test_format_session_summary_matches_group_format():
    session = _build_session(review_notes="Looks safe")
    lines = format_session_summary(session)
    assert any("Notes: Looks safe" in line for line in lines)
    assert any("Reward audit entries:" in line for line in lines)
