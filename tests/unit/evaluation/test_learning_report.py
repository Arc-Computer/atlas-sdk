import pytest

from atlas.evaluation.learning_report import (
    LearningSummary,
    generate_learning_summary,
    summary_to_dict,
    summary_to_markdown,
)


class FakeDatabase:
    def __init__(self):
        self.trajectory_calls = []

    async def fetch_sessions_for_learning_key(self, learning_key, **_):
        assert learning_key == "demo-key"
        return [
            {
                "id": 1,
                "task": "Summarise telemetry",
                "status": "succeeded",
                "review_status": "approved",
                "metadata": {
                    "execution_mode": "coach",
                    "adaptive_summary": {"adaptive_mode": "coach"},
                },
                "reward": {"score": 0.92},
                "reward_stats": {"score": 0.9, "uncertainty_mean": 0.1},
                "reward_audit": [{"stage": "tier1"}],
                "student_learning": "Capture better telemetry refs.",
                "teacher_learning": None,
                "created_at": None,
                "completed_at": None,
            },
            {
                "id": 2,
                "task": "Summarise telemetry",
                "status": "succeeded",
                "review_status": "approved",
                "metadata": {
                    "adaptive_summary": {"adaptive_mode": "paired"},
                },
                "reward": {"score": 0.88, "uncertainty": 0.2},
                "reward_stats": {"score": 0.88},
                "reward_audit": [],
                "student_learning": None,
                "teacher_learning": "Focus on alignment validation.",
                "created_at": None,
                "completed_at": None,
            },
        ]

    async def fetch_reward_baseline(self, learning_key, window):
        assert learning_key == "demo-key"
        assert window == 20
        return {"score_mean": 0.8, "sample_count": 4}

    async def fetch_discovery_runs(self, **_):
        return [
            {"id": 5, "task": "Summarise telemetry", "source": "discovery", "created_at": None},
            {"id": 7, "task": "Summarise telemetry", "source": "runtime", "created_at": None},
        ]

    async def fetch_trajectory_events(self, session_id, limit):
        self.trajectory_calls.append((session_id, limit))
        return [{"id": 1}, {"id": 2}]


@pytest.mark.asyncio
async def test_generate_learning_summary_computes_fields():
    database = FakeDatabase()
    summary = await generate_learning_summary(
        database,
        "demo-key",
        recent_window=2,
        baseline_window=20,
        discovery_limit=3,
        trajectory_limit=50,
    )
    assert isinstance(summary, LearningSummary)
    assert summary.learning_key == "demo-key"
    assert summary.session_count == 2
    assert summary.reward.recent_mean is not None
    assert summary.reward.baseline_mean == pytest.approx(0.8)
    assert summary.reward.delta == pytest.approx(summary.reward.recent_mean - 0.8)
    assert summary.adaptive_modes == {"coach": 1, "paired": 1}
    assert summary.discovery_runs[0].run_id == 5
    assert database.trajectory_calls == [(1, 50), (2, 50)]
    payload = summary_to_dict(summary)
    assert payload["learning_key"] == "demo-key"
    markdown = summary_to_markdown(summary)
    assert "Learning Evaluation" in markdown
    assert "reward delta" in markdown.lower()
