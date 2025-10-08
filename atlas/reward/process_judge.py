"""Deprecated shim for :mod:`atlas.evaluation.judges.process`."""

from __future__ import annotations

import warnings

from atlas.evaluation.judges.process import ProcessJudge  # noqa: F401

warnings.warn(
    "atlas.reward.process_judge is deprecated; import from atlas.evaluation.judges.process",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["ProcessJudge"]
