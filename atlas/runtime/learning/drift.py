"""Utilities for detecting reward drift before traces enter training."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Dict, Optional, Sequence, TYPE_CHECKING

MAD_NORMALIZATION = 1.4826


if TYPE_CHECKING:  # pragma: no cover - for typing only
    from atlas.runtime.storage.database import Database


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive guard
        return None


def _zscore(value: Optional[float], mean: Optional[float], stddev: Optional[float]) -> Optional[float]:
    if value is None or mean is None or stddev is None or stddev == 0:
        return None
    return (value - mean) / stddev


def _mad_zscore(value: Optional[float], samples: Sequence[float] | None) -> Optional[float]:
    if value is None or not samples:
        return None
    sample_values = [float(sample) for sample in samples]
    if not sample_values:
        return None
    median_value = median(sample_values)
    deviations = [abs(sample - median_value) for sample in sample_values]
    mad = median(deviations)
    if mad == 0:
        return None
    # Normalize MAD to approximate standard deviation scaling.
    return (value - median_value) / (mad * MAD_NORMALIZATION)


def _delta(value: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if value is None or baseline is None:
        return None
    return value - baseline


def _sanitize_baseline(baseline: Dict[str, Any]) -> Dict[str, Any]:
    keys = {
        "window",
        "sample_count",
        "score_mean",
        "score_median",
        "score_stddev",
        "uncertainty_mean",
        "uncertainty_median",
        "uncertainty_stddev",
        "best_uncertainty_mean",
        "best_uncertainty_median",
        "best_uncertainty_stddev",
    }
    return {key: baseline.get(key) for key in keys}


@dataclass(slots=True)
class DriftAssessment:
    """Result of evaluating a session against historical reward baselines."""

    alert: bool
    reason: Optional[str]
    score_delta: Optional[float]
    score_z: Optional[float]
    score_mad_z: Optional[float]
    uncertainty_delta: Optional[float]
    uncertainty_z: Optional[float]
    uncertainty_mad_z: Optional[float]
    threshold: float
    window: int
    learning_key: Optional[str]
    baseline: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert": self.alert,
            "drift_alert": self.alert,
            "reason": self.reason,
            "score_delta": self.score_delta,
            "score_z": self.score_z,
            "score_mad_z": self.score_mad_z,
            "uncertainty_delta": self.uncertainty_delta,
            "uncertainty_z": self.uncertainty_z,
            "uncertainty_mad_z": self.uncertainty_mad_z,
            "threshold": self.threshold,
            "window": self.window,
            "learning_key": self.learning_key,
            "baseline": _sanitize_baseline(self.baseline),
        }


class RewardDriftDetector:
    """Computes simple z-score or MAD-based drift alerts for reward signals."""

    def __init__(
        self,
        *,
        window: int = 50,
        z_threshold: float = 3.0,
        min_baseline: int = 5,
    ) -> None:
        self._window = max(window, 1)
        self._threshold = max(z_threshold, 0.0)
        self._min_baseline = max(min_baseline, 0)

    @property
    def window(self) -> int:
        return self._window

    @property
    def threshold(self) -> float:
        return self._threshold

    async def assess(
        self,
        database: "Database",
        *,
        learning_key: Optional[str],
        current_stats: Optional[Dict[str, Any]],
    ) -> Optional[DriftAssessment]:
        if not isinstance(current_stats, dict):
            return None

        baseline = await database.fetch_reward_baseline(learning_key, window=self._window)
        sample_count = int(baseline.get("sample_count") or 0)

        score_value = _coerce_float(current_stats.get("score"))
        if score_value is None:
            score_value = _coerce_float(current_stats.get("score_mean"))
        uncertainty_value = _coerce_float(
            current_stats.get("uncertainty_mean")
            if current_stats.get("uncertainty_mean") is not None
            else current_stats.get("best_uncertainty")
        )

        score_delta = _delta(score_value, _coerce_float(baseline.get("score_mean")))
        score_z = _zscore(score_value, _coerce_float(baseline.get("score_mean")), _coerce_float(baseline.get("score_stddev")))
        score_mad_z = _mad_zscore(score_value, baseline.get("scores"))

        uncertainty_baseline = _coerce_float(
            baseline.get("uncertainty_mean")
            if baseline.get("uncertainty_mean") is not None
            else baseline.get("best_uncertainty_mean")
        )
        uncertainty_stddev = _coerce_float(
            baseline.get("uncertainty_stddev")
            if baseline.get("uncertainty_stddev") is not None
            else baseline.get("best_uncertainty_stddev")
        )
        uncertainty_samples = baseline.get("uncertainties") or baseline.get("best_uncertainties")
        uncertainty_delta = _delta(uncertainty_value, uncertainty_baseline)
        uncertainty_z = _zscore(uncertainty_value, uncertainty_baseline, uncertainty_stddev)
        uncertainty_mad_z = _mad_zscore(uncertainty_value, uncertainty_samples)

        reason: Optional[str] = None
        alert = False

        if sample_count >= self._min_baseline:
            for metric_name, metric_value in (
                ("score_z", score_z),
                ("score_mad_z", score_mad_z),
                ("uncertainty_z", uncertainty_z),
                ("uncertainty_mad_z", uncertainty_mad_z),
            ):
                if metric_value is not None and abs(metric_value) >= self._threshold:
                    alert = True
                    reason = metric_name
                    break
        else:
            reason = "insufficient_baseline"

        return DriftAssessment(
            alert=alert,
            reason=reason,
            score_delta=score_delta,
            score_z=score_z,
            score_mad_z=score_mad_z,
            uncertainty_delta=uncertainty_delta,
            uncertainty_z=uncertainty_z,
            uncertainty_mad_z=uncertainty_mad_z,
            threshold=self._threshold,
            window=self._window,
            learning_key=learning_key,
            baseline=baseline,
        )


__all__ = [
    "DriftAssessment",
    "RewardDriftDetector",
]
