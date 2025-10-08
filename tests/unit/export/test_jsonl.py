import json
from pathlib import Path

import pytest

from atlas.cli.jsonl_writer import DEFAULT_TRAJECTORY_LIMIT, ExportRequest, export_sessions_sync


class FakeDatabase:
    def __init__(self, _config):
        self.connected = False

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.connected = False

    async def fetch_sessions(self, limit: int = 50, offset: int = 0):
        return [
            {
                "id": 1,
                "task": "demo task",
                "status": "succeeded",
                "metadata": {"dataset": "demo"},
                "created_at": None,
                "completed_at": None,
            }
        ]

    async def fetch_session(self, session_id: int):
        assert session_id == 1
        return {
            "id": 1,
            "task": "demo task",
            "status": "succeeded",
            "metadata": {"dataset": "demo"},
            "final_answer": "done",
            "plan": {
                "steps": [
                    {
                        "id": 1,
                        "description": "collect data",
                        "tool": "search",
                        "tool_params": {"query": "atlas"},
                        "depends_on": [],
                    }
                ]
            },
            "created_at": None,
            "completed_at": None,
        }

    async def fetch_session_steps(self, session_id: int):
        assert session_id == 1
        return [
            {
                "step_id": 1,
                "trace": "AI: finished",
                "output": "atlas summary",
                "evaluation": {
                    "validation": {"valid": True, "rationale": "looks good"},
                    "reward": {
                        "score": 0.92,
                        "judges": [
                            {
                                "identifier": "process",
                                "score": 0.9,
                                "rationale": "solid reasoning",
                                "principles": [],
                                "samples": [
                                    {
                                        "score": 0.9,
                                        "rationale": "good",
                                        "principles": [],
                                        "uncertainty": 0.1,
                                        "temperature": 0.2,
                                    }
                                ],
                                "escalated": False,
                                "escalation_reason": None,
                            }
                        ],
                    },
                },
                "attempts": 1,
                "metadata": {
                    "reasoning": [
                        {
                            "message_index": 2,
                            "role": "ai",
                            "payload": {"reasoning_content": [{"type": "thought", "text": "analysis"}]},
                        }
                    ]
                },
                "guidance_notes": ["cite sources"],
                "attempt_details": [{"attempt": 1, "evaluation": {"reward": {"score": 0.9}}}],
            }
        ]

    async def fetch_trajectory_events(self, session_id: int, limit: int = DEFAULT_TRAJECTORY_LIMIT):
        assert session_id == 1
        assert limit == DEFAULT_TRAJECTORY_LIMIT
        return [
            {
                "id": 99,
                "event": {"type": "TASK_START", "name": "step_1"},
                "created_at": None,
            }
        ]


class EmptyDatabase(FakeDatabase):
    async def fetch_sessions(self, limit: int = 50, offset: int = 0):
        return []

    async def fetch_session(self, session_id: int):
        return None

    async def fetch_session_steps(self, session_id: int):
        return []

    async def fetch_trajectory_events(self, session_id: int, limit: int = 200):
        return []


def test_exporter_writes_expected_jsonl(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("atlas.cli.jsonl_writer.Database", FakeDatabase)
    output_path = tmp_path / "traces.jsonl"

    request = ExportRequest(
        database_url="postgresql://stub",
        output_path=output_path,
        limit=1,
    )

    summary = export_sessions_sync(request)
    assert summary.sessions == 1
    assert summary.steps == 1
    payloads = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(payloads) == 1
    record = payloads[0]
    assert record["task"] == "demo task"
    assert record["final_answer"] == "done"
    assert isinstance(record["plan"], dict) and record["plan"]["steps"][0]["description"] == "collect data"
    assert record["session_metadata"]["status"] == "succeeded"
    assert record["trajectory_events"][0]["type"] == "TASK_START"
    step = record["steps"][0]
    assert step["description"] == "collect data"
    assert step["reward"]["score"] == pytest.approx(0.92)
    assert step["reward"]["judges"][0]["identifier"] == "process"
    assert step["validation"]["valid"] is True
    assert step["guidance"] == ["cite sources"]
    assert step["metadata"]["reasoning"][0]["payload"]["reasoning_content"][0]["text"] == "analysis"
    assert step["metadata"]["attempt_history"][0]["attempt"] == 1


def test_exporter_handles_empty_results(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("atlas.cli.jsonl_writer.Database", EmptyDatabase)
    output_path = tmp_path / "traces.jsonl"

    request = ExportRequest(
        database_url="postgresql://stub",
        output_path=output_path,
        limit=5,
    )

    summary = export_sessions_sync(request)
    assert summary.sessions == 0
    assert summary.steps == 0
    assert output_path.read_text(encoding="utf-8") == ""
