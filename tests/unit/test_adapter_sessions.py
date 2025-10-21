import asyncio
import asyncio
import sys
from types import ModuleType

import pytest

from atlas.connectors.python import PythonAdapter
from atlas.connectors.registry import AgentAdapter
from atlas.connectors.registry import SessionContext
from atlas.connectors.registry import StatelessSession
from atlas.config.models import AdapterConfig
from atlas.config.models import AdapterType
from atlas.config.models import PythonAdapterConfig
from atlas.config.models import StudentConfig
from atlas.config.models import StudentPrompts
from atlas.personas.student import Student
from atlas.prompts import build_student_prompts
from atlas.runtime.orchestration.execution_context import ExecutionContext


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class _EchoAdapter(AgentAdapter):
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def ainvoke(self, prompt: str, metadata: dict | None = None) -> str:
        self.calls.append(dict(metadata or {}))
        return prompt.upper()


def test_stateless_session_injects_metadata():
    adapter = _EchoAdapter()
    context = SessionContext(task_id="task-1", execution_mode="plan")
    session = StatelessSession(adapter, context)

    result = asyncio.run(session.step("hello", {}))

    assert result == "HELLO"
    assert adapter.calls, "adapter should have been invoked"
    assert "adapter_session_id" in adapter.calls[0]


def test_python_adapter_stateful_hooks(monkeypatch):
    module = ModuleType("_stateful_adapter_module")

    class StatefulAgent:
        def __init__(self) -> None:
            self.counter = 0
            self.closed_with: str | None = None

        async def on_open(self, context: SessionContext) -> dict[str, str]:
            return {"opened": context.task_id}

        async def on_step(self, prompt: str, metadata: dict | None = None) -> dict:
            self.counter += 1
            return {"content": prompt, "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}

        async def on_close(self, reason: str | None = None) -> None:
            self.closed_with = reason

    module.StatefulAgent = StatefulAgent
    monkeypatch.setitem(sys.modules, module.__name__, module)

    adapter_config = PythonAdapterConfig(
        type=AdapterType.PYTHON,
        name="stateful",
        system_prompt="",
        import_path=module.__name__,
        attribute="StatefulAgent",
    )
    adapter = PythonAdapter(adapter_config)
    assert adapter.supports_sessions is True

    session_context = SessionContext(task_id="unit", execution_mode="test")
    session = asyncio.run(adapter.open_session(session_context))
    assert session.metadata.get("opened") == "unit"

    first = asyncio.run(session.step("ping", {}))
    second = asyncio.run(session.step("pong", {}))
    assert first["content"] == "ping"
    assert second["content"] == "pong"
    asyncio.run(session.close(reason="finished"))


@pytest.mark.anyio("asyncio")
async def test_student_session_scope_tracks_usage():
    class StubSession:
        def __init__(self) -> None:
            self.session_id = "stub-session"
            self.metadata = {"stub": True}
            self.calls: list[str] = []

        async def step(self, payload: str, metadata: dict | None = None) -> dict[str, object]:
            self.calls.append(payload)
            if len(self.calls) == 1:
                return {
                    "content": "{\"steps\":[{\"id\":1,\"description\":\"Do\",\"tool\":null,\"tool_params\":null,\"depends_on\":[]}]}",
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    "events": [{"type": "completion", "stage": "plan"}],
                }
            return {
                "content": "Final answer",
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

        async def close(self, reason: str | None = None) -> dict[str, object]:
            return {"closed": reason}

    class StubAdapter(AgentAdapter):
        supports_sessions = True

        def __init__(self) -> None:
            self.session = StubSession()

        async def ainvoke(self, prompt: str, metadata: dict | None = None) -> str:
            return prompt

        async def open_session(self, context: SessionContext):
            return self.session

    ExecutionContext.get().reset()
    adapter = StubAdapter()
    adapter_config = AdapterConfig(
        type=AdapterType.PYTHON,
        name="stub",
        system_prompt="",
    )
    student_config = StudentConfig(
        prompts=StudentPrompts(
            planner="Plan",
            executor="Execute",
            synthesizer="Synthesize",
        )
    )
    student_prompts = build_student_prompts(adapter_config.system_prompt, student_config)
    student = Student(adapter, adapter_config, student_config, student_prompts)

    task = "Test"
    async with student.session_scope(task, "stepwise"):
        plan = await student.acreate_plan(task)
        assert plan.steps
        answer = await student.asynthesize_final_answer(task, [])
        assert answer == "Final answer"

    session_meta = ExecutionContext.get().metadata.get("adapter_session")
    assert isinstance(session_meta, dict)
    assert session_meta.get("adapter_session_id") == "stub-session"
    usage_meta = session_meta.get("usage")
    assert isinstance(usage_meta, dict)
    assert usage_meta.get("calls", 0) >= 2
    events_meta = session_meta.get("events")
    assert isinstance(events_meta, list)

