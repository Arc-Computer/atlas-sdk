"""Deprecated shim for :mod:`atlas.evaluation.judges.base`."""

from __future__ import annotations

import warnings

from atlas.evaluation.judges.base import (
    Judge,
    JudgeContext,
    JudgeOutcome,
    JudgeSample,
)

warnings.warn(
    "atlas.reward.judge is deprecated; import from atlas.evaluation.judges.base",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["Judge", "JudgeContext", "JudgeOutcome", "JudgeSample"]
