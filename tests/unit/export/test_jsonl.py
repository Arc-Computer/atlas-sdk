import json
from pathlib import Path
from types import SimpleNamespace

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
                "review_status": "approved",
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
            "review_status": "approved",
            "metadata": {
                "dataset": "demo",
                "adaptive_summary": {"adaptive_mode": "coach", "confidence": 0.72},
                "learning_history": {"count": 2, "average_score": 0.85},
                "learning_key": "demo-learning-key",
                "reward_summary": {"average": 0.88, "count": 1},
                "student_learning": "Refine executive tone.",
                "teacher_learning": "Flag missing citations.",
            },
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
            "reward": json.dumps({"score": 0.91, "rationale": "Judge average"}),
            "reward_stats": {"score": 0.91, "score_stddev": 0.04, "sample_count": 3},
            "reward_audit": [
                {
                    "stage": "tier1",
                    "model": "demo-model",
                    "temperature": 0.2,
                    "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
                    "response": "{\"score\": 0.9}",
                }
            ],
            "student_learning": "Refine executive tone.",
            "teacher_learning": "Flag missing citations.",
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
                    "validation": {"valid": True, "guidance": "looks good"},
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
                "event": {
                    "event_type": "TASK_START",
                    "name": "step_1",
                    "metadata": {"actor": "student"},
                },
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
    assert record["review_status"] == "approved"
    assert record["execution_mode"] == "coach"
    assert record["reward_stats"]["score"] == pytest.approx(0.91)
    assert record["session_metadata"]["status"] == "succeeded"
    assert record["adaptive_summary"]["adaptive_mode"] == "coach"
    assert record.get("learning_history", {}).get("count") == 2
    assert record.get("learning_key") == "demo-learning-key"
    assert record["session_reward"]["score"] == pytest.approx(0.91)
    assert record["reward_audit"][0]["stage"] == "tier1"
    assert record["student_learning"] == "Refine executive tone."
    assert record["teacher_learning"] == "Flag missing citations."
    event_entry = record["trajectory_events"][0]
    assert event_entry["type"] == "TASK_START"
    assert event_entry["actor"] == "student"
    assert event_entry["event"]["metadata"]["actor"] == "student"
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


def test_run_export_uses_config_defaults(monkeypatch, tmp_path: Path):
    from atlas.cli import export as export_cli

    monkeypatch.delenv("ATLAS_REVIEW_DEFAULT_EXPORT_STATUSES", raising=False)
    monkeypatch.delenv("ATLAS_REVIEW_REQUIRE_APPROVAL", raising=False)

    captured_request: SimpleNamespace | None = None

    def fake_export(request):
        nonlocal captured_request
        captured_request = request
        return SimpleNamespace(sessions=1, steps=10)

    monkeypatch.setattr(export_cli, "export_sessions_sync", fake_export)
    monkeypatch.setattr(export_cli, "configure_logging", lambda quiet: None)
    monkeypatch.setattr(
        "atlas.config.loader.load_config",
        lambda path: SimpleNamespace(
            runtime_safety=SimpleNamespace(
                review=SimpleNamespace(require_approval=False, default_export_statuses=["approved", "pending"])
            )
        ),
    )

    exit_code = export_cli._run_export(
        [
            "--config",
            "configs/runtime.yaml",
            "--database-url",
            "postgresql://stub",
            "--output",
            str(tmp_path / "traces.jsonl"),
            "--quiet",
        ]
    )

    assert exit_code == 0
    assert captured_request is not None
    assert captured_request.review_status_filters == ["approved", "pending"]
    assert captured_request.include_all_review_statuses is False
