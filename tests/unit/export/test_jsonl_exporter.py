import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from atlas.export.jsonl import DEFAULT_TRAJECTORY_LIMIT, ExportStats, export_sessions


class FakeStore:
    def __init__(self, sessions, session_details, steps, events):
        self._sessions = sessions
        self._session_details = session_details
        self._steps = steps
        self._events = events

    async def fetch_sessions(self, limit=50, offset=0):
        return self._sessions[offset : offset + limit]

    async def fetch_session(self, session_id):
        return self._session_details.get(session_id)

    async def fetch_session_steps(self, session_id):
        return list(self._steps.get(session_id, []))

    async def fetch_trajectory_events(self, session_id, limit=DEFAULT_TRAJECTORY_LIMIT):
        return list(self._events.get(session_id, []))[:limit]


@pytest.fixture
def populated_store():
    plan = {
        "steps": [
            {
                "id": 1,
                "description": "Collect supporting documents",
                "tool": "document_tool",
                "tool_params": {"query": "atlas"},
                "depends_on": [],
            },
            {
                "id": 2,
                "description": "Summarise findings",
                "tool": "summariser",
                "tool_params": {"style": "concise"},
                "depends_on": [1],
            },
        ]
    }
    validation_payload = {"valid": True, "rationale": "Looks good"}
    reward_payload = {
        "score": 0.91,
        "judges": [
            {
                "identifier": "process",
                "score": 0.9,
                "rationale": "Solid reasoning",
                "principles": [{"id": "accuracy"}],
                "samples": [
                    {
                        "score": 0.9,
                        "rationale": "Consistent outputs",
                        "principles": [],
                        "uncertainty": 0.1,
                        "temperature": 0.2,
                    }
                ],
                "escalated": False,
                "escalation_reason": None,
            }
        ],
    }
    second_reward = dict(reward_payload, score=0.82)

    sessions = [
        {
            "id": 1,
            "task": "Process paperwork",
            "status": "succeeded",
            "metadata": json.dumps({"source": "unit-test"}),
            "final_answer": "Completed",
            "created_at": datetime(2025, 1, 2, 12, 0, tzinfo=timezone.utc),
            "completed_at": datetime(2025, 1, 2, 12, 5, tzinfo=timezone.utc),
        }
    ]
    session_details = {
        1: {
            **sessions[0],
            "plan": json.dumps(plan),
        }
    }
    steps = {
        1: [
            {
                "step_id": 1,
                "trace": "trace-1",
                "output": "documents collected",
                "evaluation": json.dumps({"validation": validation_payload, "reward": reward_payload}),
                "attempts": 1,
                "attempt_details": [
                    {"attempt": 1, "evaluation": json.dumps({"validation": validation_payload, "reward": reward_payload})}
                ],
                "guidance_notes": ["double-check receipts"],
            },
            {
                "step_id": 2,
                "trace": "trace-2",
                "output": "summary ready",
                "evaluation": json.dumps({"validation": validation_payload, "reward": second_reward}),
                "attempts": 2,
                "attempt_details": [
                    {"attempt": 1, "evaluation": json.dumps({"validation": validation_payload, "reward": reward_payload})},
                    {"attempt": 2, "evaluation": json.dumps({"validation": validation_payload, "reward": second_reward})},
                ],
                "guidance_notes": ["focus on key points", "wrap up"],
            },
        ]
    }
    events = {
        1: [
            {
                "id": 5,
                "event": json.dumps({"payload": "end"}),
                "created_at": datetime(2025, 1, 2, 12, 5, tzinfo=timezone.utc),
            },
            {
                "id": 4,
                "event": json.dumps({"payload": "start"}),
                "created_at": datetime(2025, 1, 2, 12, 0, tzinfo=timezone.utc),
            },
        ]
    }
    return FakeStore(sessions, session_details, steps, events)


def test_export_sessions_writes_expected_payload(populated_store, tmp_path):
    output_path = tmp_path / "traces.jsonl"
    stats = asyncio.run(export_sessions(populated_store, output_path))

    assert isinstance(stats, ExportStats)
    assert stats.sessions == 1
    assert stats.steps == 2

    contents = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(contents) == 1
    record = json.loads(contents[0])

    assert record["task"] == "Process paperwork"
    assert record["plan"]["steps"][0]["description"] == "Collect supporting documents"
    assert record["steps"][0]["description"] == "Collect supporting documents"
    assert record["steps"][0]["reward"]["score"] == pytest.approx(0.91)
    assert record["steps"][1]["context"]["prior_results"]["1"] == "documents collected"
    assert record["steps"][1]["depends_on"] == [1]
    assert record["steps"][1]["attempt_history"][1]["evaluation"]["reward"]["score"] == pytest.approx(0.82)
    assert record["steps"][1]["guidance"] == ["focus on key points", "wrap up"]

    events = record["session_metadata"]["trajectory_events"]
    assert [event["id"] for event in events] == [4, 5]
    assert record["session_metadata"]["status"] == "succeeded"
    assert "created_at" in record["session_metadata"]


def test_export_sessions_handles_empty_result(tmp_path):
    store = FakeStore([], {}, {}, {})
    output_path = tmp_path / "empty.jsonl"

    stats = asyncio.run(export_sessions(store, output_path))

    assert stats.sessions == 0
    assert stats.steps == 0
    assert output_path.read_text(encoding="utf-8") == ""


def test_export_sessions_filters_session_ids(populated_store, tmp_path):
    # Add a second session that should be ignored when filtering
    populated_store._sessions.append(
        {
            "id": 2,
            "task": "Secondary",
            "status": "succeeded",
            "metadata": None,
            "final_answer": "",
            "created_at": datetime.now(timezone.utc),
            "completed_at": datetime.now(timezone.utc),
        }
    )
    cloned_detail = dict(populated_store._session_details[1])
    cloned_detail["id"] = 2
    populated_store._session_details[2] = cloned_detail
    populated_store._steps[2] = populated_store._steps[1]
    populated_store._events[2] = populated_store._events[1]

    output_path = tmp_path / "filtered.jsonl"
    stats = asyncio.run(export_sessions(populated_store, output_path, session_ids=[2]))

    assert stats.sessions == 1
    record = json.loads(output_path.read_text(encoding="utf-8").strip())
    assert record["session_metadata"]["session_id"] == 2
