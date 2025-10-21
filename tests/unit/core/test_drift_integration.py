import pytest

from atlas.core import _persist_results
from atlas.config.models import RuntimeSafetyConfig
from atlas.runtime.orchestration.execution_context import ExecutionContext, ExecutionContextState
from atlas.runtime.schema import AtlasRewardBreakdown
from atlas.types import Plan, Result


class DriftAwareStubDatabase:
    def __init__(self):
        self.logged_reward = None
        self.logged_stats = None
        self.logged_audit = None
        self.metadata_updates = []
        self.review_updates = []

    async def log_plan(self, session_id, plan):
        return None

    async def log_step_result(self, session_id, step_result):
        return None

    async def log_step_attempts(self, session_id, step_id, attempts):
        return None

    async def log_guidance(self, session_id, step_id, guidance):
        return None

    async def log_session_reward(self, session_id, reward, student_learning, teacher_learning, reward_stats, reward_audit):
        self.logged_reward = reward
        self.logged_stats = reward_stats
        self.logged_audit = reward_audit

    async def update_session_metadata(self, session_id, metadata):
        self.metadata_updates.append(metadata)

    async def log_intermediate_step(self, session_id, event):
        return None

    async def update_session_review_status(self, session_id, review_status, notes=None):
        self.review_updates.append((session_id, review_status, notes))

    async def fetch_reward_baseline(self, learning_key=None, window=50):
        return {
            "sample_count": 10,
            "score_mean": 0.55,
            "score_stddev": 0.03,
            "scores": [0.54, 0.55, 0.56, 0.55, 0.54, 0.55, 0.56, 0.55, 0.55, 0.56],
            "uncertainty_mean": 0.08,
            "uncertainty_stddev": 0.005,
            "uncertainties": [0.08 for _ in range(10)],
            "best_uncertainty_mean": 0.07,
            "best_uncertainty_stddev": 0.004,
            "best_uncertainties": [0.07 for _ in range(10)],
        }


@pytest.mark.asyncio
async def test_persist_results_flags_drift_and_updates_metadata(monkeypatch):
    database = DriftAwareStubDatabase()
    context = ExecutionContext(ExecutionContextState.get())
    context.reset()
    context.metadata["steps"] = {}
    context.metadata["learning_key"] = "demo"
    context.metadata["session_metadata"] = {}
    reward = AtlasRewardBreakdown(score=0.92, judges=[], rationale=None)
    context.metadata["session_reward"] = reward
    context.metadata["session_reward_stats"] = {
        "score": 0.92,
        "score_stddev": 0.01,
        "sample_count": 3,
        "uncertainty_mean": 0.12,
    }

    plan = Plan(steps=[])
    result = Result(final_answer="ok", plan=plan, step_results=[])

    await _persist_results(
        database,
        session_id=99,
        context=context,
        result=result,
        events=[],
        runtime_safety=RuntimeSafetyConfig(),
    )

    # Reward stats persisted with timestamp and drift triggered review update.
    assert database.logged_stats is not None
    assert "timestamp" in database.logged_stats
    assert database.logged_audit is None
    assert database.review_updates == [(99, "pending", "Reward drift alert (score_z)")]

    # Metadata now includes drift payload with alert flag.
    merged_metadata = database.metadata_updates[-1]
    assert merged_metadata["drift"]["drift_alert"] is True
    assert merged_metadata["drift"]["score_delta"] is not None


@pytest.mark.asyncio
async def test_persist_results_auto_approves_when_review_disabled(monkeypatch):
    database = DriftAwareStubDatabase()
    context = ExecutionContext(ExecutionContextState.get())
    context.reset()
    context.metadata["steps"] = {}
    context.metadata["session_metadata"] = {}
    plan = Plan(steps=[])
    result = Result(final_answer="ok", plan=plan, step_results=[])
    runtime_safety = RuntimeSafetyConfig()
    runtime_safety.review.require_approval = False
    runtime_safety.review.default_export_statuses = ["approved", "pending"]
    runtime_safety.drift.enabled = False

    await _persist_results(
        database,
        session_id=101,
        context=context,
        result=result,
        events=[],
        runtime_safety=runtime_safety,
    )

    assert database.review_updates == [(101, "approved", "Auto-approved (review gating disabled).")]
    merged_metadata = database.metadata_updates[-1]
    assert merged_metadata["review_status"] == "approved"
    assert merged_metadata["review"]["require_approval"] is False
    assert merged_metadata["review"]["default_export_statuses"] == ["approved", "pending"]
    assert merged_metadata["review"]["include_all_statuses"] is False
