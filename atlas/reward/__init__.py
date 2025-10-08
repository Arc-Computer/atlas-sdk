"""Deprecated reward module."""

from __future__ import annotations

import warnings

from atlas.evaluation.evaluator import Evaluator
from atlas.evaluation.judges.base import JudgeContext, JudgeOutcome, JudgeSample
from atlas.evaluation.judges.helpfulness import HelpfulnessJudge
from atlas.evaluation.judges.process import ProcessJudge

warnings.warn(
    "atlas.reward is deprecated; import from atlas.evaluation",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "Evaluator",
    "HelpfulnessJudge",
    "JudgeContext",
    "JudgeOutcome",
    "JudgeSample",
    "ProcessJudge",
]
