"""Deprecated orchestration shim."""

from __future__ import annotations

import warnings

from atlas.runtime.orchestration.orchestrator import Orchestrator
from atlas.runtime.orchestration.execution_context import ExecutionContext
from atlas.runtime.orchestration.dependency_graph import DependencyGraph
from atlas.runtime.orchestration.step_manager import IntermediateStepManager

warnings.warn(
    "atlas.orchestration is deprecated; import from atlas.runtime.orchestration",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "DependencyGraph",
    "ExecutionContext",
    "IntermediateStepManager",
    "Orchestrator",
]
