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
        self.event_count_session_ids = None

    async def fetch_learning_sessions(self, *, learning_key=None, project_root=None, task=None, tags=None, limit=None, order=None):
        assert learning_key == "demo-key"
        assert project_root is None
        assert task is None
        assert tags in (None, [], ())
        assert limit is None or limit >= 0
        assert order in (None, "asc", "desc")
        return [
            {
                "id": 1,
                "task": "Summarise telemetry",
                "status": "succeeded",
                "review_status": "approved",
                "metadata": {
                    "execution_mode": "coach",
                    "adaptive_summary": {"adaptive_mode": "coach"},
                    "adapter_session": {"student_model_id": "student-alpha", "teacher_model_id": "teacher-prime"},
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
                    "adapter_session": {"student_model_id": "student-beta", "teacher_model_id": "teacher-prime"},
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

    async def fetch_trajectory_event_counts(self, session_ids):
        self.event_count_session_ids = list(session_ids)
        return {sid: 2 for sid in session_ids}


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
    assert summary.reward.recent_window.size == 2
    assert summary.reward.baseline_window.size == 20
    assert summary.adaptive_modes == {"coach": 1, "paired": 1}
    assert summary.discovery_runs[0].run_id == 5
    assert database.trajectory_calls == [(1, 50), (2, 50)]
    assert database.event_count_session_ids is None
    assert {entry.role for entry in summary.model_breakdown} == {"student", "teacher"}
    student_models = {entry.model_id: entry.session_count for entry in summary.model_breakdown if entry.role == "student"}
    assert student_models == {"student-alpha": 1, "student-beta": 1}
    teacher_models = {entry.model_id: entry.session_count for entry in summary.model_breakdown if entry.role == "teacher"}
    assert teacher_models == {"teacher-prime": 2}
    latest_snapshot = summary.sessions[-1]
    assert latest_snapshot.student_model_id == "student-beta"
    assert latest_snapshot.teacher_model_id == "teacher-prime"
    payload = summary_to_dict(summary)
    assert payload["learning_key"] == "demo-key"
    markdown = summary_to_markdown(summary)
    assert "Learning Evaluation" in markdown
    assert "reward delta" in markdown.lower()
    assert "Model Performance" in markdown
    assert "student-beta" in markdown
    assert "student learning: Capture better telemetry refs." in markdown


@pytest.mark.asyncio
async def test_generate_learning_summary_summary_only_uses_event_counts():
    database = FakeDatabase()
    summary = await generate_learning_summary(
        database,
        "demo-key",
        recent_window=2,
        baseline_window=20,
        summary_only=True,
    )
    assert database.event_count_session_ids == [1, 2]
    assert database.trajectory_calls == []
    assert all(snapshot.trajectory_events == 2 for snapshot in summary.sessions)
