from __future__ import annotations

import math
from pathlib import Path

import pytest

from atlas.config.loader import load_config
from atlas.evaluation import RewardEvaluation
from atlas.runtime.schema import AtlasJudgeBreakdown, AtlasJudgeSample, AtlasRewardBreakdown

from scripts.eval_reward_models import (
    JUDGE_COMBOS,
    TrajectoryRecord,
    aggregate_results,
    evaluate_combo,
    load_reward_dataset,
    _build_rim_config,
)


def test_load_reward_dataset_shapes():
    records = load_reward_dataset(path=Path("atlas/data/reward_eval_trajectories.jsonl"))
    assert len(records) >= 6
    first = records[0]
    assert isinstance(first, TrajectoryRecord)
    assert first.trajectory.task.startswith("Investigate marketing landing page")
    assert first.trajectory.execution_mode == "paired"
    assert len(first.trajectory.steps) == 3
    warm = next(record for record in records if record.trajectory.execution_mode == "auto")
    assert warm.trajectory_type == "warm_start"
    assert warm.trajectory.teacher_intervened is False


def test_build_rim_config_applies_presets():
    config = load_config("configs/examples/openai_agent.yaml")
    combo = JUDGE_COMBOS["claude_stack"]
    rim_config = _build_rim_config(config.rim, combo)
    assert rim_config.small_model.provider.value == "anthropic"
    assert rim_config.small_model.model == "claude-haiku-4-5"
    assert rim_config.large_model.provider.value == "anthropic"
    assert rim_config.large_model.model == "claude-sonnet-4-5-20250929"


def test_build_rim_config_handles_new_presets():
    config = load_config("configs/examples/openai_agent.yaml")

    gpt_combo = JUDGE_COMBOS["gpt5_stack"]
    gpt_rim = _build_rim_config(config.rim, gpt_combo)
    assert gpt_rim.small_model.provider.value == "openai"
    assert gpt_rim.small_model.model == "gpt-5-mini"
    assert gpt_rim.large_model.provider.value == "openai"
    assert gpt_rim.large_model.model == "gpt-5"

    grok_combo = JUDGE_COMBOS["grok_stack"]
    grok_rim = _build_rim_config(config.rim, grok_combo)
    assert grok_rim.small_model.provider.value == "xai"
    assert grok_rim.small_model.model == "xai/grok-4-fast"
    assert grok_rim.large_model.provider.value == "xai"
    assert grok_rim.large_model.model == "xai/grok-4"


class _StubEvaluator:
    def __init__(self) -> None:
        self._calls = 0

    async def aevaluate_session(self, trajectory):
        self._calls += 1
        base_score = 0.6 + 0.01 * self._calls
        uncertainty = 0.05 + 0.01 * (self._calls % 3)
        sample = AtlasJudgeSample(
            score=base_score,
            rationale="stub rationale",
            principles=[],
            uncertainty=uncertainty,
            temperature=None,
        )
        judge = AtlasJudgeBreakdown(
            identifier="session_reward",
            score=base_score,
            rationale="stub",
            principles=[],
            samples=[sample],
            escalated=(self._calls % 2 == 0),
        )
        reward = AtlasRewardBreakdown(
            score=base_score,
            judges=[judge],
            rationale=None,
            raw={"samples": [{"score": base_score, "uncertainty": uncertainty}]},
        )
        return RewardEvaluation(
            reward=reward,
            student_learning="keep improving",
            teacher_learning=None,
        )


@pytest.mark.asyncio
async def test_evaluate_combo_and_aggregate_with_stub(monkeypatch):
    records = load_reward_dataset(Path("atlas/data/reward_eval_trajectories.jsonl"))[:2]
    config = load_config("configs/examples/openai_agent.yaml")
    combo = JUDGE_COMBOS["gemini_pair"]
    rim_config = _build_rim_config(config.rim, combo)

    async def _run():
        return await evaluate_combo(
            combo,
            rim_config,
            records,
            repeats=2,
            concurrency=1,
            evaluator_factory=lambda _: _StubEvaluator(),
        )

    per_run = await _run()
    assert len(per_run) == len(records) * 2
    assert all(entry["score"] is not None for entry in per_run)
    latencies = [entry["latency_ms"] for entry in per_run]
    assert all(latency >= 0 for latency in latencies)

    # Use the same data for both baseline and comparison to exercise aggregation logic.
    aggregated = aggregate_results(
        per_run,
        baseline_combo="gemini_pair",
    )
    summary = aggregated["gemini_pair"]
    assert summary["runs"] == len(per_run)
    expected_scores = [0.6 + 0.01 * (index + 1) for index in range(len(per_run))]
    assert math.isclose(summary["score_mean"], sum(expected_scores) / len(expected_scores), rel_tol=1e-6)
    assert summary["failures"] == 0
    assert summary["escalation_rate"] == 0.5
    assert summary["agreement"]["samples"] == len(per_run)
    assert math.isclose(summary["agreement"]["mean_delta"], 0.0, abs_tol=1e-9)
    assert math.isclose(summary["agreement"]["within_0_02"], 1.0, abs_tol=1e-9)
    assert math.isclose(summary["agreement"]["pearson"], 1.0, rel_tol=1e-6)
