"""Unit tests for training_data converters."""

import json
from typing import Any, Dict, List

import pytest

from atlas.runtime.schema import AtlasSessionTrace, AtlasStepTrace
from atlas.training_data.converters import convert_session_dict_to_trace, convert_step_dict_to_trace


def test_session_conversion_preserves_essential_fields():
    """Verify all 6 essential fields are populated correctly."""
    session_dict = {
        "id": 1,
        "task": "test task",
        "final_answer": "test answer",
        "plan": {"steps": [{"id": 1, "description": "step 1"}]},
        "reward": json.dumps({"score": 0.9, "rationale": "good"}),
        "student_learning": "student cue",
        "teacher_learning": "teacher cue",
        "metadata": {
            "learning_history": {"count": 2},
            "adaptive_summary": {"mode": "coach"},
        },
    }
    steps = [
        {
            "step_id": 1,
            "trace": "trace",
            "output": "output",
            "evaluation": {"reward": {"score": 0.8}},
            "attempts": 1,
            "metadata": {},
            "guidance_notes": [],
        }
    ]
    trajectory_events = [{"id": 1, "event": {"type": "test"}, "created_at": None}]

    trace = convert_session_dict_to_trace(session_dict, steps, trajectory_events, include_learning_data=True)

    assert isinstance(trace, AtlasSessionTrace)
    assert trace.task == "test task"
    assert trace.final_answer == "test answer"
    assert trace.session_reward is not None
    assert trace.session_reward.get("score") == 0.9
    assert trace.student_learning == "student cue"
    assert trace.teacher_learning == "teacher cue"
    assert trace.learning_history == {"count": 2}
    assert trace.adaptive_summary == {"mode": "coach"}
    assert trace.trajectory_events is not None
    assert len(trace.trajectory_events) == 1


def test_property_accessors():
    """Verify @property accessors work correctly."""
    session_dict = {
        "id": 1,
        "task": "test",
        "final_answer": "answer",
        "plan": {"steps": []},
        "metadata": {
            "learning_key": "task-1",
            "teacher_notes": ["note1", "note2"],
            "reward_summary": {"avg": 0.8},
            "drift": {"detected": True},
            "drift_alert": True,
            "triage_dossier": {"category": "bug"},
            "reward_audit": [{"stage": "tier1"}],
        },
    }

    trace = convert_session_dict_to_trace(session_dict, [], [], include_learning_data=True)

    assert trace.learning_key == "task-1"
    assert trace.teacher_notes == ["note1", "note2"]
    assert trace.reward_summary == {"avg": 0.8}
    assert trace.drift == {"detected": True}
    assert trace.drift_alert is True
    assert trace.triage_dossier == {"category": "bug"}
    assert trace.reward_audit == [{"stage": "tier1"}]


def test_backward_compatibility_missing_fields():
    """Verify old data without new fields doesn't crash."""
    session_dict = {
        "id": 1,
        "task": "test",
        "final_answer": "answer",
        "plan": {"steps": []},
        "metadata": {},
    }

    trace = convert_session_dict_to_trace(session_dict, [], [], include_learning_data=True)

    assert trace.session_reward is None
    assert trace.student_learning is None
    assert trace.teacher_learning is None
    assert trace.learning_history is None
    assert trace.adaptive_summary is None
    assert trace.trajectory_events is None


def test_step_conversion_preserves_fields():
    """Verify step fields are populated correctly."""
    step_dict = {
        "step_id": 1,
        "trace": "trace text",
        "output": "output text",
        "evaluation": {
            "reward": {"score": 0.9, "judges": []},
            "validation": {"valid": True},
        },
        "attempts": 2,
        "metadata": {
            "runtime": {"duration": 1.5},
            "artifacts": {"file": "test.txt"},
            "deliverable": "result",
        },
        "guidance_notes": ["guidance"],
    }
    plan = {
        "steps": [
            {
                "id": 1,
                "description": "step description",
                "tool": "test_tool",
                "tool_params": {"param": "value"},
                "depends_on": [0],
            }
        ]
    }

    trace = convert_step_dict_to_trace(step_dict, plan)

    assert isinstance(trace, AtlasStepTrace)
    assert trace.step_id == 1
    assert trace.description == "step description"
    assert trace.runtime == {"duration": 1.5}
    assert trace.depends_on == [0]
    assert trace.artifacts == {"file": "test.txt"}
    assert trace.deliverable == "result"
    assert trace.attempt_history is not None


def test_step_property_accessor():
    """Verify attempt_history property works."""
    step_dict = {
        "step_id": 1,
        "trace": "",
        "output": "",
        "evaluation": {"reward": {"score": 0.0}},
        "attempts": 1,
        "metadata": {
            "attempt_history": [
                {"attempt": 1, "evaluation": {"reward": {"score": 0.8}}},
                {"attempt": 2, "evaluation": {"reward": {"score": 0.9}}},
            ]
        },
        "guidance_notes": [],
    }
    plan = {"steps": [{"id": 1, "description": "test"}]}

    trace = convert_step_dict_to_trace(step_dict, plan)

    assert trace.attempt_history is not None
    assert len(trace.attempt_history) == 2
    assert trace.attempt_history[0]["attempt"] == 1


def test_selective_loading():
    """Verify include_learning_data=False skips learning fields."""
    session_dict = {
        "id": 1,
        "task": "test",
        "final_answer": "answer",
        "plan": {"steps": []},
        "reward": json.dumps({"score": 0.9}),
        "student_learning": "cue",
        "metadata": {"learning_history": {"count": 1}},
    }

    trace = convert_session_dict_to_trace(session_dict, [], [], include_learning_data=False)

    assert trace.session_reward is None
    assert trace.student_learning is None
    assert trace.learning_history is None

