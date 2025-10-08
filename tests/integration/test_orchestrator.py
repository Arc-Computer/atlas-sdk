import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, ToolMessage
from atlas.config.models import OrchestrationConfig, RIMConfig, LLMParameters
from atlas.runtime.orchestration.execution_context import ExecutionContext
from atlas.runtime.orchestration.orchestrator import Orchestrator
from atlas.runtime.models import IntermediateStepType
from atlas.evaluation.judges.base import JudgeContext
from atlas.runtime.schema import AtlasRewardBreakdown
from atlas.personas.student import StudentStepResult
from atlas.types import Plan, Result, Step


class FakeStudent:
    def __init__(self):
        self._attempts: dict[int, int] = {}

    async def acreate_plan(self, task: str) -> Plan:
        return Plan(
            steps=[
                Step(id=1, description="gather dataset A", depends_on=[]),
                Step(id=2, description="gather dataset B", depends_on=[]),
                Step(id=3, description="synthesize findings", depends_on=[1, 2]),
            ]
        )

    async def aexecute_step(self, step: Step, context, guidance):
        attempts = self._attempts.get(step.id, 0) + 1
        self._attempts[step.id] = attempts
        if step.id == 1:
            output = "success-a" if attempts >= 2 else "pending-a"
            tool_id = "call-1"
        elif step.id == 2:
            output = "success-b"
            tool_id = "call-2"
        else:
            part_a = context.get(1, "")
            part_b = context.get(2, "")
            output = f"summary:{part_a}|{part_b}"
            tool_id = "call-3"
        tool_calls = [{"name": "search", "args": {"query": step.description}, "id": tool_id}]
        messages = [
            AIMessage(content="calling tool", tool_calls=tool_calls),
            ToolMessage(content='{"result": 1}', tool_call_id=tool_id),
        ]
        return StudentStepResult(trace="trace", output=output, messages=messages, attempts=attempts)

    async def asynthesize_final_answer(self, task: str, step_summaries):
        outputs = [entry["output"] for entry in step_summaries]
        return " | ".join(outputs)


class FakeTeacher:
    async def areview_plan(self, task: str, plan: Plan) -> Plan:
        return plan

    async def avalidate_step(self, step: Step, trace: str, output: str):
        if step.id == 1:
            return {"valid": output.startswith("success"), "rationale": ""}
        if step.id == 2:
            return {"valid": output.startswith("success"), "rationale": ""}
        return {"valid": output.startswith("summary:"), "rationale": ""}

    async def agenerate_guidance(self, step: Step, evaluation):
        return "retry with missing references"

    def collect_results(self, items):
        return sorted(items, key=lambda payload: payload["step_id"])


class FakeEvaluator:
    async def ajudge(self, context: JudgeContext):
        output = context.output or ""
        if output.startswith("success") or output.startswith("summary"):
            score = 0.9
        else:
            score = 0.2
        return AtlasRewardBreakdown(score=score, judges=[], raw={"score": score})


def build_orchestrator():
    orchestration_config = OrchestrationConfig(max_retries=1, step_timeout_seconds=900, rim_guidance_tag="tag", emit_intermediate_steps=True)
    rim_config = RIMConfig(
        small_model=LLMParameters(model="stub"),
        large_model=LLMParameters(model="arbiter"),
        active_judges={"process": True, "helpfulness": True},
        variance_threshold=1.0,
        uncertainty_threshold=1.0,
        parallel_workers=2,
    )
    return Orchestrator(
        teacher=FakeTeacher(),
        student=FakeStudent(),
        evaluator=FakeEvaluator(),
        orchestration_config=orchestration_config,
        rim_config=rim_config,
    )


@pytest.mark.asyncio
async def test_orchestrator_retries_and_records_context():
    context = ExecutionContext.get()
    context.reset()
    events = []
    subscription = context.event_stream.subscribe(events.append)
    orchestrator = build_orchestrator()
    try:
        result = await orchestrator.arun("task")
    finally:
        subscription.unsubscribe()
    assert isinstance(result, Result)
    step_attempts = {entry.step_id: entry.attempts for entry in result.step_results}
    assert step_attempts[1] == 2
    assert step_attempts[2] == 1
    assert step_attempts[3] == 1
    step_meta = context.metadata["steps"][1]
    event_types = [event.event_type for event in events]
    assert IntermediateStepType.WORKFLOW_START in event_types
    assert IntermediateStepType.TASK_START in event_types
    assert IntermediateStepType.TASK_END in event_types
    assert len(step_meta["attempts"]) == 2
    assert step_meta["guidance"] == ["retry with missing references"]
