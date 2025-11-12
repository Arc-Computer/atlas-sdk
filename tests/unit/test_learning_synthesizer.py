from __future__ import annotations

import json

import pytest

from atlas.config.models import LearningConfig, PlaybookEntrySchemaConfig
from atlas.learning.schema import build_playbook_entry_schema
from atlas.learning.synthesizer import LearningSynthesizer


class _StubResponse:
    def __init__(self, content: str) -> None:
        self.content = content
        self.raw = {"raw": True}
        self.reasoning = {"trace": ["step"]}


class _StubLLMClient:
    def __init__(self, payload: str, model: str = "stub-learning-model") -> None:
        self.model = model
        self._payload = payload
        self.calls = 0
        self.messages = None
        self.response_format = None
        self.overrides = None

    async def acomplete(self, messages, response_format=None, overrides=None):
        self.calls += 1
        self.messages = messages
        self.response_format = response_format
        self.overrides = overrides
        return _StubResponse(self._payload)


@pytest.mark.asyncio
async def test_learning_synthesizer_merges_state_and_trims_history():
    cfg = LearningConfig(
        enabled=True,
        update_enabled=True,
        llm=None,
        history_limit=2,
        schema=PlaybookEntrySchemaConfig(allowed_runtime_handles=["validate_assumptions"]),
    )
    existing_state = {
        "student_learning": "Existing guidance block.",
        "teacher_learning": None,
        "metadata": {
            "version": 1,
            "playbook_entries": [
                {
                    "id": "legacy",
                    "audience": "student",
                    "scope": {"category": "reinforcement"},
                    "provenance": {"status": {"category": "reinforcement", "lifecycle": "active"}},
                }
            ],
        },
    }
    history = {
        "entries": [{"id": 1}, {"id": 2}, {"id": 3}],
        "count": 3,
    }
    response_payload = json.dumps(
        {
            "version": "playbook_entry.v1",
            "student_pamphlet": "Existing guidance block.\n- Validate assumptions early.",
            "teacher_pamphlet": "Prompt them to enumerate blockers.",
            "session_student_learning": "Focus on validating assumptions before acting.",
            "session_teacher_learning": "Prompt the student to list blocking factors.",
            "playbook_entries": [
                {
                    "id": "entry-validate",
                    "audience": "student",
                    "cue": {
                        "type": "keyword",
                        "pattern": "validate assumptions",
                        "description": "When the task mentions assumptions",
                    },
                    "action": {
                        "imperative": "Run the assumption validation checklist",
                        "runtime_handle": "validate_assumptions",
                        "tool_name": "validate_assumptions",
                        "arguments": {},
                    },
                    "expected_effect": "Avoid chasing incorrect hypotheses.",
                    "scope": {
                        "category": "reinforcement",
                        "constraints": "General troubleshooting",
                    },
                    "metadata": {},
                }
            ],
            "metadata": {"version": 2},
        }
    )
    client = _StubLLMClient(response_payload)
    synthesizer = LearningSynthesizer(cfg, client=client)
    result = await synthesizer.asynthesize(
        learning_key="tenant::agent",
        task="Diagnose failing batch job",
        reward={"score": 0.82},
        trajectory={"steps": [{"description": "Investigate logs"}]},
        learning_state=existing_state,
        history=history,
    )
    assert client.calls == 1
    assert isinstance(result.student_learning, str)
    assert result.student_learning.startswith("Focus on validating")
    assert result.teacher_learning == "Prompt the student to list blocking factors."
    assert result.learning_state["student_learning"].startswith("Existing guidance block.")
    assert result.learning_state["teacher_learning"] == "Prompt them to enumerate blockers."
    metadata = result.learning_state["metadata"]
    assert metadata["version"] == 2
    assert "policy_nuggets" not in metadata
    playbook_entries = metadata.get("playbook_entries")
    assert playbook_entries
    statuses = {entry["id"]: entry["provenance"]["status"]["lifecycle"] for entry in playbook_entries}
    assert statuses["entry-validate"] == "active"
    assert statuses.get("legacy") == "deprecated"
    assert result.playbook_entries is not None
    active_entries = [entry for entry in result.playbook_entries if entry["provenance"]["status"]["lifecycle"] == "active"]
    assert len(active_entries) == 1
    assert result.rubric_summary is not None and result.rubric_summary.get("accepted") is True
    assert result.session_note == "Student: Focus on validating assumptions before acting. Teacher: Prompt the student to list blocking factors."
    message_payload = json.loads(client.messages[-1]["content"])
    assert "pamphlets" in message_payload
    trimmed_entries = message_payload["history"]["entries"]
    assert len(trimmed_entries) == 2
    assert trimmed_entries[0]["id"] == 2 and trimmed_entries[1]["id"] == 3


@pytest.mark.asyncio
async def test_learning_synthesizer_skips_when_updates_disabled():
    cfg = LearningConfig(enabled=True, update_enabled=False, llm=None)
    client = _StubLLMClient("{}")
    synthesizer = LearningSynthesizer(cfg, client=client)
    result = await synthesizer.asynthesize(
        learning_key="tenant::agent",
        task="Any task",
        reward=None,
        trajectory=None,
        learning_state={},
        history=None,
    )
    assert result is None
    assert client.calls == 0


@pytest.mark.asyncio
async def test_learning_synthesizer_rejects_on_gate_failure():
    cfg = LearningConfig(
        enabled=True,
        update_enabled=True,
        llm=None,
        schema=PlaybookEntrySchemaConfig(allowed_runtime_handles=["valid_handle"]),
    )
    baseline_state = {
        "student_learning": "Keep queries focused.",
        "metadata": {
            "playbook_entries": [
                {
                    "id": "existing",
                    "audience": "student",
                    "scope": {"category": "reinforcement"},
                    "provenance": {"status": {"category": "reinforcement", "lifecycle": "active"}},
                }
            ]
        },
    }
    response_payload = json.dumps(
        {
            "version": "playbook_entry.v1",
            "student_pamphlet": "Replace with risky guidance",
            "playbook_entries": [
                {
                    "audience": "student",
                    "cue": {"type": "keyword", "pattern": "incident 5"},
                    "action": {"imperative": "Do something specific", "runtime_handle": "unknown_handle"},
                    "expected_effect": "",
                    "scope": {"category": "differentiation", "constraints": "Incident #5"},
                }
            ],
        }
    )
    client = _StubLLMClient(response_payload)
    synthesizer = LearningSynthesizer(cfg, client=client)
    result = await synthesizer.asynthesize(
        learning_key="tenant::agent",
        task="Diagnose",
        reward=None,
        trajectory=None,
        learning_state=baseline_state,
        history=None,
    )
    assert result.learning_state["student_learning"] == "Keep queries focused."
    metadata = result.learning_state["metadata"]
    assert "policy_nuggets" not in metadata
    assert metadata["playbook_entries"][0]["provenance"]["status"]["lifecycle"] == "active"
    assert metadata["last_failure"]["rejected_candidates"]
    assert result.playbook_entries[0]["provenance"]["status"]["lifecycle"] == "active"
    assert result.rubric_summary["accepted"] is False


def test_build_playbook_entry_schema():
    """Test that JSON schema is correctly generated."""
    schema = build_playbook_entry_schema()
    
    assert schema["type"] == "object"
    assert "version" in schema["properties"]
    assert schema["properties"]["version"]["const"] == "playbook_entry.v1"
    assert "playbook_entries" in schema["properties"]
    assert schema["properties"]["playbook_entries"]["type"] == "array"
    
    # Check playbook entry structure
    entry_schema = schema["properties"]["playbook_entries"]["items"]
    assert "audience" in entry_schema["properties"]
    assert entry_schema["properties"]["audience"]["enum"] == ["student", "teacher"]
    assert "cue" in entry_schema["properties"]
    assert "action" in entry_schema["properties"]
    assert "scope" in entry_schema["properties"]
    assert entry_schema["properties"]["scope"]["properties"]["category"]["enum"] == ["reinforcement", "differentiation"]


@pytest.mark.asyncio
async def test_learning_synthesizer_uses_structured_outputs_for_gemini():
    """Test that Gemini models use structured outputs with JSON schema."""
    cfg = LearningConfig(
        enabled=True,
        update_enabled=True,
        llm=None,
        schema=PlaybookEntrySchemaConfig(allowed_runtime_handles=["test_handle"]),
    )
    response_payload = json.dumps({
        "version": "playbook_entry.v1",
        "student_pamphlet": None,
        "teacher_pamphlet": None,
        "playbook_entries": [],
        "session_student_learning": None,
        "session_teacher_learning": None,
        "metadata": None
    })
    
    # Test with Gemini model
    client = _StubLLMClient(response_payload, model="gemini/gemini-2.5-flash")
    synthesizer = LearningSynthesizer(cfg, client=client)
    result = await synthesizer.asynthesize(
        learning_key="test::agent",
        task="Test task",
        reward={"score": 0.8},
        trajectory=None,
        learning_state={},
        history=None,
    )
    
    assert client.calls == 1
    assert client.response_format == {"type": "json_object"}
    assert client.overrides is not None
    assert "extra_body" in client.overrides
    assert "response_json_schema" in client.overrides["extra_body"]
    assert client.overrides["extra_body"]["response_json_schema"]["type"] == "object"
    assert result is not None
    assert result.audit is not None
    assert result.audit.get("structured_output") is True


@pytest.mark.asyncio
async def test_learning_synthesizer_backward_compatible_non_gemini():
    """Test that non-Gemini models still work without structured outputs."""
    cfg = LearningConfig(
        enabled=True,
        update_enabled=True,
        llm=None,
        schema=PlaybookEntrySchemaConfig(allowed_runtime_handles=["test_handle"]),
    )
    response_payload = json.dumps({
        "version": "playbook_entry.v1",
        "student_pamphlet": None,
        "teacher_pamphlet": None,
        "playbook_entries": [],
        "session_student_learning": None,
        "session_teacher_learning": None,
        "metadata": None
    })
    
    # Test with OpenAI model
    client = _StubLLMClient(response_payload, model="gpt-4")
    synthesizer = LearningSynthesizer(cfg, client=client)
    result = await synthesizer.asynthesize(
        learning_key="test::agent",
        task="Test task",
        reward={"score": 0.8},
        trajectory=None,
        learning_state={},
        history=None,
    )
    
    assert client.calls == 1
    assert client.response_format == {"type": "json_object"}
    # Non-Gemini models should not have structured outputs
    assert client.overrides is None or "extra_body" not in client.overrides or "response_json_schema" not in client.overrides.get("extra_body", {})
    assert result is not None
    assert result.audit is not None
    assert result.audit.get("structured_output") is False


def test_is_gemini_model_detection():
    """Test Gemini model detection."""
    synthesizer = LearningSynthesizer(LearningConfig(enabled=False))
    
    assert synthesizer._is_gemini_model("gemini/gemini-2.5-flash") is True
    assert synthesizer._is_gemini_model("gemini/gemini-2.5-pro") is True
    assert synthesizer._is_gemini_model("google/gemini-pro") is True
    assert synthesizer._is_gemini_model("gpt-4") is False
    assert synthesizer._is_gemini_model("claude-3-opus") is False
    assert synthesizer._is_gemini_model("") is False
