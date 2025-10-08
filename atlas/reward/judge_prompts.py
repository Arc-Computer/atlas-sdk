"""Deprecated shim for :mod:`atlas.evaluation.judges.prompts`."""

from __future__ import annotations

import warnings

from atlas.evaluation.judges.prompts import *  # noqa: F401,F403

warnings.warn(
    "atlas.reward.judge_prompts is deprecated; import from atlas.evaluation.judges.prompts",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [name for name in globals() if name.isupper() or name.endswith("PROMPT")]
