"""Deprecated shim for :mod:`atlas.evaluation.evaluator`."""

from __future__ import annotations

import warnings

from atlas.evaluation.evaluator import Evaluator  # noqa: F401

warnings.warn(
    "atlas.reward.evaluator is deprecated; import from atlas.evaluation.evaluator",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["Evaluator"]
