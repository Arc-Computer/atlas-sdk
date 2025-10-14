"""Utilities for building deterministic persona fingerprints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Sequence

from atlas.config.models import AtlasConfig, ToolDefinition
from atlas.runtime.orchestration.execution_context import ExecutionContext


@dataclass(frozen=True)
class FingerprintInputs:
    """Input parameters used to build a persona fingerprint."""

    agent_name: str
    tenant_id: str
    tools: Sequence[str]
    task_tags: Sequence[str]


def _sorted_names(items: Sequence[ToolDefinition]) -> list[str]:
    return sorted({tool.name for tool in items if getattr(tool, "name", None)})


def build_fingerprint(inputs: FingerprintInputs) -> str:
    """Return a deterministic hash for a persona fingerprint."""
    payload = {
        "agent_name": inputs.agent_name,
        "tenant_id": inputs.tenant_id,
        "tools": sorted(inputs.tools),
        "task_tags": sorted(inputs.task_tags),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def extract_fingerprint_inputs(task: str, config: AtlasConfig, context: ExecutionContext) -> FingerprintInputs:
    """Derive fingerprint inputs from the current task, configuration, and execution context."""
    del task  # placeholder for future task-derived features
    metadata = context.metadata
    session_metadata = metadata.get("session_metadata") or {}
    tenant_id = session_metadata.get("tenant_id", "default")
    tools_field: Sequence[ToolDefinition] | None = getattr(config.agent, "tools", None)
    tools = _sorted_names(tools_field or [])
    raw_tags: Any = session_metadata.get("tags") or []
    if isinstance(raw_tags, (list, tuple, set)):
        tags = [str(tag) for tag in raw_tags]
    elif raw_tags:
        tags = [str(raw_tags)]
    else:
        tags = []
    return FingerprintInputs(
        agent_name=config.agent.name,
        tenant_id=tenant_id,
        tools=tuple(tools),
        task_tags=tuple(sorted(tags)),
    )
