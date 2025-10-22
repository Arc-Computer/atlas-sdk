from __future__ import annotations

import json

import pytest

from atlas.config.models import LearningConfig
from atlas.learning.synthesizer import LearningSynthesizer


class _StubResponse:
    def __init__(self, content: str) -> None:
        self.content = content
        self.raw = {"raw": True}
        self.reasoning = {"trace": ["step"]}


class _StubLLMClient:
    def __init__(self, payload: str) -> None:
        self.model = "stub-learning-model"
        self._payload = payload
        self.calls = 0
        self.messages = None

    async def acomplete(self, messages, response_format=None, overrides=None):
        self.calls += 1
        self.messages = messages
        return _StubResponse(self._payload)


@pytest.mark.asyncio
async def test_learning_synthesizer_merges_state_and_trims_history():
    cfg = LearningConfig(enabled=True, update_enabled=True, llm=None, history_limit=2)
    existing_state = {
        "student_learning": "Existing guidance block.",
        "teacher_learning": None,
        "metadata": {"version": 1},
    }
    history = {
        "entries": [{"id": 1}, {"id": 2}, {"id": 3}],
        "count": 3,
    }
    response_payload = json.dumps(
        {
            "student_pamphlet": "Existing guidance block.\n- Validate assumptions early.",
            "teacher_pamphlet": "Prompt them to enumerate blockers.",
            "session_student_learning": "Focus on validating assumptions before acting.",
            "session_teacher_learning": "Prompt the student to list blocking factors.",
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
    assert result.learning_state["metadata"]["version"] == 2
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
