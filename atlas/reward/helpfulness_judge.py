"""Deprecated shim for :mod:`atlas.evaluation.judges.helpfulness`."""

from __future__ import annotations

import warnings

from atlas.evaluation.judges.helpfulness import HelpfulnessJudge  # noqa: F401

warnings.warn(
    "atlas.reward.helpfulness_judge is deprecated; import from atlas.evaluation.judges.helpfulness",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["HelpfulnessJudge"]
