import json
import sys
import types
import pytest

if "langchain_core" not in sys.modules:
    langchain_core = types.ModuleType("langchain_core")
    langchain_core.__path__ = []  # mark as package

    messages_module = types.ModuleType("langchain_core.messages")
    for name in ("BaseMessage", "AIMessage", "HumanMessage", "SystemMessage", "ToolMessage"):
        setattr(messages_module, name, type(name, (), {}))

    messages_tool_module = types.ModuleType("langchain_core.messages.tool")
    messages_tool_module.ToolCall = type("ToolCall", (), {})

    language_models_module = types.ModuleType("langchain_core.language_models")
    language_models_module.BaseChatModel = type("BaseChatModel", (), {"__init__": lambda self, *a, **k: None})

    outputs_module = types.ModuleType("langchain_core.outputs")
    outputs_module.ChatGeneration = type("ChatGeneration", (), {"__init__": lambda self, *a, **k: None})
    outputs_module.ChatResult = type("ChatResult", (), {"__init__": lambda self, *a, **k: None})

    tools_module = types.ModuleType("langchain_core.tools")

    class _BaseTool:
        pass

    class _StructuredTool:
        @classmethod
        def from_function(cls, func, coroutine=None, name=None, description=None, args_schema=None):
            instance = cls()
            instance.func = func
            instance.coroutine = coroutine
            instance.name = name
            instance.description = description
            instance.args_schema = args_schema
            return instance

    tools_module.BaseTool = _BaseTool
    tools_module.StructuredTool = _StructuredTool

    langchain_core.messages = messages_module
    langchain_core.language_models = language_models_module
    langchain_core.outputs = outputs_module
    langchain_core.tools = tools_module
    sys.modules["langchain_core"] = langchain_core
    sys.modules["langchain_core.messages"] = messages_module
    sys.modules["langchain_core.messages.tool"] = messages_tool_module
    sys.modules["langchain_core.language_models"] = language_models_module
    sys.modules["langchain_core.outputs"] = outputs_module
    sys.modules["langchain_core.tools"] = tools_module

from atlas.config.models import LearningConfig, LLMParameters, LLMProvider
from atlas.evaluation.evaluator import SessionTrajectory
from atlas.learning.synthesizer import LearningSynthesizer
from atlas.runtime.schema import AtlasJudgeBreakdown, AtlasRewardBreakdown


def _llm_params() -> LLMParameters:
    return LLMParameters(provider=LLMProvider.OPENAI, model="gpt-test")


def _trajectory() -> SessionTrajectory:
    return SessionTrajectory(
        task="demo-task",
        final_answer="done",
        plan={},
        steps=[],
        execution_mode="auto",
        teacher_intervened=False,
        session_metadata=None,
        focus_prompt=None,
    )


def _reward() -> AtlasRewardBreakdown:
    judge = AtlasJudgeBreakdown(
        identifier="session_reward",
        score=0.5,
        rationale="baseline",
        principles=[],
        samples=[],
        escalated=False,
        escalation_reason=None,
    )
    return AtlasRewardBreakdown(
        score=0.5,
        judges=[judge],
        rationale=None,
        raw={"score": 0.5},
    )


class RecordingClient:
    def __init__(self, *_args, **_kwargs) -> None:
        self.called = False

    async def acomplete(self, *_args, **_kwargs):
        self.called = True
        raise AssertionError("acomplete should not be invoked when updates are disabled")


class StubDatabase:
    def __init__(self) -> None:
        self.last_upsert = None

    async def upsert_learning_state(self, learning_key, student, teacher, metadata):
        self.last_upsert = (learning_key, student, teacher, metadata)

    async def fetch_learning_state(self, learning_key):
        if self.last_upsert is None:
            return None
        return {
            "learning_key": learning_key,
            "student_learning": self.last_upsert[1],
            "teacher_learning": self.last_upsert[2],
            "metadata": self.last_upsert[3],
            "updated_at": "2024-01-01T00:00:00",
        }


@pytest.mark.asyncio
async def test_learning_synthesizer_disabled_returns_prior_state(monkeypatch):
    config = LearningConfig(llm=_llm_params(), updates_enabled=False)
    monkeypatch.setattr("atlas.learning.synthesizer.LLMClient", RecordingClient)
    synthesizer = LearningSynthesizer(config, None)
    prior_state = {
        "learning_key": "alpha",
        "student_learning": "keep direct answer generation when facts are complete.",
        "teacher_learning": "skip intervention unless validation fails.",
        "metadata": {"source": "seed"},
        "updated_at": "2024-01-01T00:00:00",
    }
    history = {"entries": [], "count": 0}
    update = await synthesizer.asynthesize(
        learning_key="alpha",
        trajectory=_trajectory(),
        reward=_reward(),
        history_snapshot=history,
        prior_state=prior_state,
    )
    assert update.session_student_learning == prior_state["student_learning"]
    assert update.session_teacher_learning == prior_state["teacher_learning"]
    assert update.updated_state == prior_state
    assert update.history_snapshot["count"] == 0
    assert update.history_snapshot["current_student_learning"] == prior_state["student_learning"]


class SuccessClient:
    last_messages = None

    def __init__(self, *_args, **_kwargs) -> None:
        SuccessClient.last_messages = []

    async def acomplete(self, messages, response_format=None):
        SuccessClient.last_messages = messages
        payload = {
            "student_pamphlet": "Prefer structured validation before answering.",
            "teacher_pamphlet": None,
            "session_student_learning": "Prefer structured validation before answering.",
            "session_teacher_learning": None,
            "metadata": {"reason": "high variance in tool success"},
        }
        content = json.dumps(payload)
        return type("Resp", (), {"content": content})


@pytest.mark.asyncio
async def test_learning_synthesizer_updates_state(monkeypatch):
    config = LearningConfig(llm=_llm_params(), updates_enabled=True, max_history_entries=1)
    db = StubDatabase()
    monkeypatch.setattr("atlas.learning.synthesizer.LLMClient", SuccessClient)
    synthesizer = LearningSynthesizer(config, db)
    history = {
        "entries": [
            {
                "reward": {"score": 0.3, "rationale": "slow"},
                "student_learning": "Avoid redundant retries.",
                "teacher_learning": None,
                "created_at": "2023-12-01T00:00:00",
                "completed_at": "2023-12-01T00:00:10",
            },
            {
                "reward": {"score": 0.8, "rationale": "fast"},
                "student_learning": "Cache expensive lookups.",
                "teacher_learning": "Signal when cache misses.",
                "created_at": "2023-12-02T00:00:00",
                "completed_at": "2023-12-02T00:00:10",
            },
        ],
        "count": 2,
    }
    update = await synthesizer.asynthesize(
        learning_key="beta",
        trajectory=_trajectory(),
        reward=_reward(),
        history_snapshot=history,
        prior_state=None,
    )
    body = json.loads(SuccessClient.last_messages[1]["content"])
    assert len(body["history"]) == 1
    assert db.last_upsert == (
        "beta",
        "Prefer structured validation before answering.",
        None,
        {"reason": "high variance in tool success"},
    )
    assert update.session_student_learning == "Prefer structured validation before answering."
    assert update.session_teacher_learning is None
    assert update.updated_state["student_learning"] == "Prefer structured validation before answering."
    assert update.history_snapshot["count"] == 2
    assert update.history_snapshot["current_student_learning"] == "Prefer structured validation before answering."
