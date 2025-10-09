"""Deprecated data model exports."""

from __future__ import annotations

import warnings

from atlas.runtime.models.intermediate_step import (
    IntermediateStep,
    IntermediateStepPayload,
    IntermediateStepType,
    StreamEventData,
)

warnings.warn(
    "atlas.data_models is deprecated; import from atlas.runtime.models",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "IntermediateStep",
    "IntermediateStepPayload",
    "IntermediateStepType",
    "StreamEventData",
]
