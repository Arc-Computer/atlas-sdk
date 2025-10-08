"""Atlas SDK public API.

This module re-exports the most common entry points a developer needs when
embedding Atlas into their own systems:

* Runtime orchestration helpers (``run`` / ``arun``)
* Student / Teacher personas for tailoring behaviour
* Connector registry helpers for registering Bring-Your-Own-Agent adapters
* Runtime schemas used by the training stack
* Reward evaluator facade
* JSONL exporter utilities for turning persisted sessions into datasets

The re-exported surface mirrors the conceptual layers documented in the Atlas
core repository—``personas``, ``connectors``, ``runtime``, ``evaluation``, and
``cli``.
"""

from __future__ import annotations

from atlas.cli.jsonl_writer import ExportRequest, ExportSummary, export_sessions_sync
from atlas.connectors import (
    AdapterError,
    AgentAdapter,
    build_adapter,
    create_adapter,
    create_from_atlas_config,
    register_adapter,
)
from atlas.core import arun, run
from atlas.evaluation import Evaluator
from atlas.personas import Student, StudentStepResult, Teacher
from atlas.runtime import AtlasRewardBreakdown, AtlasSessionTrace, AtlasStepTrace

__all__ = [
    # Runtime entry points
    "run",
    "arun",
    # Personas
    "Student",
    "StudentStepResult",
    "Teacher",
    # Connector registry helpers
    "AdapterError",
    "AgentAdapter",
    "build_adapter",
    "create_adapter",
    "create_from_atlas_config",
    "register_adapter",
    # Runtime schemas
    "AtlasRewardBreakdown",
    "AtlasSessionTrace",
    "AtlasStepTrace",
    # Evaluation facade
    "Evaluator",
    # Exporter utilities
    "ExportRequest",
    "ExportSummary",
    "export_sessions_sync",
]
