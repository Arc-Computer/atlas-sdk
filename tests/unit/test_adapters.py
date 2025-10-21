import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("langchain_core")

from atlas.connectors import AdapterCapabilities, AgentAdapter
from atlas.config.models import AdapterType, PythonAdapterConfig, StudentConfig
from atlas.core import _open_adapter_session
from atlas.personas.student import Student
from atlas.prompts import build_student_prompts
from atlas.runtime.models import IntermediateStepType
from atlas.runtime.orchestration.execution_context import ExecutionContext
from atlas.runtime.orchestration.orchestrator import AdaptiveModeDecision, Orchestrator
from atlas.config.models import OrchestrationConfig, RIMConfig, LLMParameters
from atlas.cli.main import _cmd_adapters_describe


class _RecordingAdapter(AgentAdapter):
    def __init__(self) -> None:
        self.plan_calls = 0
        self.execute_calls = 0
        self.synth_calls = 0
        self.events: list[dict[str, object]] = []
        self._emitter = None

    async def aopen_session(self, *, task, metadata=None, emit_event=None):
        self._emitter = emit_event
        return AdapterCapabilities(control_loop="self", supports_stepwise=False, telemetry_stream=True)

    async def aplan(self, task: str, metadata=None):
        self.plan_calls += 1
        return {
            "steps": [
                {
                    "id": 1,
                    "description": f"Solve: {task}",
                    "tool": None,
                    "tool_params": None,
                    "depends_on": [],
                }
            ]
        }

    async def aexecute(self, task: str, plan, step, metadata=None):
        self.execute_calls += 1
        if self._emitter is not None:
            await self._emitter({"event": "progress", "payload": {"message": "executing"}})
        return {
            "trace": "adapter-trace",
            "output": "adapter-output",
            "metadata": {"status": "ok"},
            "deliverable": "adapter-output",
        }

    async def asynthesize(self, task: str, plan, step_results, metadata=None):
        self.synth_calls += 1
        return step_results[0].get("deliverable", "") if step_results else ""


@pytest.mark.asyncio
async def test_student_self_managed_hooks(monkeypatch):
    ExecutionContext.get().reset()
    adapter = _RecordingAdapter()
    adapter_config = PythonAdapterConfig(
        type=AdapterType.PYTHON,
        name="test",
        system_prompt="unused",
        tools=[],
        import_path="examples.adapters",
    )
    prompts = build_student_prompts("unused", StudentConfig())
    capabilities = AdapterCapabilities(control_loop="self", supports_stepwise=False, telemetry_stream=True)
    student = Student(adapter=adapter, adapter_config=adapter_config, student_config=StudentConfig(), student_prompts=prompts, adapter_capabilities=capabilities)

    plan = await student.acreate_plan("Summarise demo")
    assert adapter.plan_calls == 1
    assert plan.execution_mode == "single_shot"

    step = plan.steps[0]
    result = await student.aexecute_step(step, context={}, guidance=[])
    assert adapter.execute_calls == 1
    assert result.output == "adapter-output"

    final_answer = await student.asynthesize_final_answer("Summarise demo", [{"deliverable": result.deliverable}])
    assert adapter.synth_calls == 1
    assert final_answer == "adapter-output"


class _TelemetryAdapter(AgentAdapter):
    def __init__(self) -> None:
        self.emitter = None

    async def aopen_session(self, *, task, metadata=None, emit_event=None):
        self.emitter = emit_event
        return AdapterCapabilities(control_loop="self", supports_stepwise=False, telemetry_stream=True)


@pytest.mark.asyncio
async def test_adapter_event_streams_into_execution_context():
    ExecutionContext.get().reset()
    adapter = _TelemetryAdapter()
    adapter_config = PythonAdapterConfig(
        type=AdapterType.PYTHON,
        name="telemetry",
        system_prompt="unused",
        tools=[],
        import_path="examples.adapters",
    )
    context = ExecutionContext.get()
    events: list = []
    subscription = context.event_stream.subscribe(events.append)
    try:
        capabilities = await _open_adapter_session(
            adapter=adapter,
            task="demo",
            execution_context=context,
            adapter_config=adapter_config,
        )
        assert capabilities.control_loop == "self"
        assert adapter.emitter is not None
        await adapter.emitter({"event": "env_action", "payload": {"detail": "call"}})
    finally:
        subscription.unsubscribe()
    assert any(event.payload.event_type == IntermediateStepType.ADAPTER_EVENT for event in events)


def test_orchestrator_enforces_adapter_capabilities():
    ExecutionContext.get().reset()
    teacher = object()
    student = object()
    evaluator = object()
    rim = RIMConfig(
        small_model=LLMParameters(model="stub"),
        large_model=LLMParameters(model="stub"),
    )
    orchestrator = Orchestrator(
        teacher=teacher,
        student=student,
        evaluator=evaluator,
        orchestration_config=OrchestrationConfig(),
        rim_config=rim,
    )
    context = ExecutionContext.get()
    context.metadata["adapter_capabilities"] = {"control_loop": "self", "supports_stepwise": False}
    decision = AdaptiveModeDecision(mode="coach", confidence=0.42, probe=None, source="probe")
    new_decision = orchestrator._enforce_adapter_capabilities(decision, context)
    assert new_decision.mode == "paired"
    assert new_decision.confidence == decision.confidence
    assert "adapter_capabilities" in (new_decision.source or "")

    # When stepwise is supported nothing changes.
    context.metadata["adapter_capabilities"] = {"control_loop": "self", "supports_stepwise": True}
    unchanged = orchestrator._enforce_adapter_capabilities(decision, context)
    assert unchanged.mode == decision.mode


def test_cli_adapters_describe_outputs_capabilities(monkeypatch, capsys):
    capabilities = AdapterCapabilities(control_loop="self", supports_stepwise=False, telemetry_stream=True)

    class StubConfig:
        agent = SimpleNamespace(behavior=None)

    async def fake_open(*_args, **_kwargs):
        return capabilities

    monkeypatch.setattr("atlas.cli.main.load_config", lambda path: StubConfig())
    monkeypatch.setattr("atlas.cli.main.create_from_atlas_config", lambda config: object())
    monkeypatch.setattr("atlas.cli.main._open_adapter_session", fake_open)
    ExecutionContext.get().reset()
    args = SimpleNamespace(config="dummy.yaml", task="demo")
    assert _cmd_adapters_describe(args) == 0
    output = capsys.readouterr().out
    assert '"control_loop": "self"' in output
