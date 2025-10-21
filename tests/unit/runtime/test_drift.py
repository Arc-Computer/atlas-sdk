import pytest

from atlas.runtime.learning.drift import RewardDriftDetector


class BaselineStubDB:
    def __init__(self, payload):
        self._payload = payload

    async def fetch_reward_baseline(self, learning_key=None, window=50):
        return self._payload


@pytest.mark.asyncio
async def test_reward_drift_detector_flags_large_score_delta():
    baseline = {
        "sample_count": 10,
        "score_mean": 0.8,
        "score_stddev": 0.05,
        "scores": [0.78, 0.82, 0.81, 0.79, 0.8, 0.81, 0.8, 0.79, 0.82, 0.8],
        "uncertainty_mean": 0.2,
        "uncertainty_stddev": 0.01,
        "uncertainties": [0.19, 0.21, 0.2, 0.2, 0.2, 0.19, 0.21, 0.2, 0.2, 0.19],
        "best_uncertainty_mean": 0.18,
        "best_uncertainty_stddev": 0.01,
        "best_uncertainties": [0.18 for _ in range(10)],
    }
    detector = RewardDriftDetector(window=10, z_threshold=3.0, min_baseline=5)
    database = BaselineStubDB(baseline)
    stats = {"score": 1.0, "uncertainty_mean": 0.25}

    assessment = await detector.assess(database, learning_key="demo", current_stats=stats)

    assert assessment is not None
    assert assessment.alert is True
    assert assessment.reason in {"score_z", "score_mad_z"}
    assert assessment.score_delta == pytest.approx(0.2, rel=1e-3)


@pytest.mark.asyncio
async def test_reward_drift_detector_handles_insufficient_baseline():
    baseline = {
        "sample_count": 2,
        "score_mean": 0.75,
        "score_stddev": 0.0,
        "scores": [0.75, 0.75],
        "uncertainty_mean": 0.15,
        "uncertainty_stddev": 0.0,
        "uncertainties": [0.15, 0.15],
        "best_uncertainty_mean": 0.14,
        "best_uncertainty_stddev": 0.0,
        "best_uncertainties": [0.14, 0.14],
    }
    detector = RewardDriftDetector(window=10, z_threshold=3.0, min_baseline=5)
    database = BaselineStubDB(baseline)
    stats = {"score": 0.9, "uncertainty_mean": 0.2}

    assessment = await detector.assess(database, learning_key="demo", current_stats=stats)

    assert assessment is not None
    assert assessment.alert is False
    assert assessment.reason == "insufficient_baseline"
