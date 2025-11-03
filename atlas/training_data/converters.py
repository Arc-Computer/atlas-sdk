"""Convert database dicts to AtlasSessionTrace and AtlasStepTrace dataclasses."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from atlas.runtime.schema import AtlasRewardBreakdown, AtlasSessionTrace, AtlasStepTrace


def _coerce_json(value: Any) -> Any:
    """
    Handle JSONB deserialization.

    asyncpg returns JSONB as dict, but some old data is JSON string.
    """
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def convert_step_dict_to_trace(
    step_dict: Dict[str, Any],
    plan: Dict[str, Any],
) -> AtlasStepTrace:
    """
    Convert database dict to AtlasStepTrace.

    Mirrors jsonl_writer._build_step_payload() logic to ensure
    identical field preservation.

    Field Extraction:
    - runtime: from metadata["runtime"]
    - depends_on: from plan step definition (not from step_results!)
    - artifacts: from metadata["artifacts"]
    - deliverable: from metadata["deliverable"]
    """
    step_id = step_dict.get("step_id")
    if not isinstance(step_id, int):
        raise ValueError(f"Invalid step_id: {step_id}")

    # Extract metadata and evaluation
    metadata = _coerce_json(step_dict.get("metadata")) or {}
    evaluation = _coerce_json(step_dict.get("evaluation")) or {}
    reward_payload = evaluation.get("reward") or {}

    # Get plan step for depends_on and description
    plan_steps = plan.get("steps", [])
    plan_step = None
    for step in plan_steps:
        if isinstance(step, dict) and step.get("id") == step_id:
            plan_step = step
            break
        elif hasattr(step, "id") and step.id == step_id:
            plan_step = step
            break

    if plan_step is None:
        plan_step = {}
    elif hasattr(plan_step, "model_dump"):
        plan_step = plan_step.model_dump()
    elif not isinstance(plan_step, dict):
        plan_step = {}

    # Extract essential fields
    runtime = metadata.get("runtime")
    depends_on = plan_step.get("depends_on")
    artifacts = metadata.get("artifacts") or {}
    deliverable = metadata.get("deliverable")

    # Handle guidance notes
    guidance = step_dict.get("guidance_notes") or []
    if not isinstance(guidance, list):
        guidance = [str(guidance)] if guidance else []

    # Build reward breakdown
    reward_breakdown = AtlasRewardBreakdown.from_dict(reward_payload)

    # Build context (simplified for now, can be enhanced if needed)
    context: Dict[str, Any] = {}

    return AtlasStepTrace(
        step_id=step_id,
        description=plan_step.get("description", ""),
        trace=step_dict.get("trace", ""),
        output=step_dict.get("output", ""),
        reward=reward_breakdown,
        tool=plan_step.get("tool"),
        tool_params=plan_step.get("tool_params", {}),
        context=context,
        validation=evaluation.get("validation", {}),
        attempts=step_dict.get("attempts", 1),
        guidance=[str(item) for item in guidance],
        metadata=metadata,
        artifacts=artifacts,
        deliverable=deliverable,
        runtime=runtime,
        depends_on=depends_on if depends_on is not None else None,
    )


def convert_session_dict_to_trace(
    session_dict: Dict[str, Any],
    steps: List[Dict[str, Any]],
    trajectory_events: List[Dict[str, Any]],
    include_learning_data: bool = True,
) -> AtlasSessionTrace:
    """
    Convert database dict to AtlasSessionTrace.

    Mirrors jsonl_writer._assemble_session() logic to ensure
    identical field preservation.

    Field Extraction:
    - session_reward: from session_dict["reward"] (JSONB column)
    - student_learning: from session_dict["student_learning"] (TEXT column) or metadata
    - teacher_learning: from session_dict["teacher_learning"] (TEXT column) or metadata
    - adaptive_summary: from metadata["adaptive_summary"]
    - learning_history: from metadata["learning_history"]
    - trajectory_events: from trajectory_events parameter (if included)
    """
    # Extract metadata
    metadata = _coerce_json(session_dict.get("metadata")) or {}

    # Extract plan
    plan_data = session_dict.get("plan")
    plan = _coerce_json(plan_data) or {}

    # Extract essential fields
    session_reward = None
    student_learning = None
    teacher_learning = None
    learning_history = None
    adaptive_summary = None

    if include_learning_data:
        session_reward = _coerce_json(session_dict.get("reward"))
        student_learning = session_dict.get("student_learning") or metadata.get("student_learning")
        teacher_learning = session_dict.get("teacher_learning") or metadata.get("teacher_learning")
        learning_history = metadata.get("learning_history")
        adaptive_summary = metadata.get("adaptive_summary")

    # Convert steps
    step_traces = [
        convert_step_dict_to_trace(step, plan) for step in steps if isinstance(step, dict)
    ]

    return AtlasSessionTrace(
        task=session_dict.get("task", ""),
        final_answer=session_dict.get("final_answer", ""),
        plan=plan,
        steps=step_traces,
        session_metadata=metadata,
        session_reward=session_reward,
        trajectory_events=trajectory_events if trajectory_events else None,
        student_learning=student_learning,
        teacher_learning=teacher_learning,
        learning_history=learning_history,
        adaptive_summary=adaptive_summary,
    )

