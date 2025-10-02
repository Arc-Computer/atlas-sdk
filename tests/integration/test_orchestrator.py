import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, ToolMessage
from atlas.config.models import OrchestrationConfig, RIMConfig, JudgeConfig, JudgeKind, LLMParameters
from atlas.orchestration.execution_context import ExecutionContext
from atlas.orchestration.orchestrator import Orchestrator
from atlas.data_models.intermediate_step import IntermediateStepType
from atlas.reward.judge import JudgeContext
from atlas.roles.student import StudentStepResult
from atlas.types import Plan, Result, Step


class FakeStudent:
    def __init__(self):
        self._attempts = 0

    async def acreate_plan(self, task: str) -> Plan:
        return Plan(steps=[Step(id=1, description="solve", depends_on=[], estimated_time="1m")], total_estimated_time="1m")

    async def aexecute_step(self, step: Step, context, guidance):
        self._attempts += 1
        output = "success" if self._attempts >= 2 else "failure"
        tool_calls = [{"name": "search", "args": {"query": "foo"}, "id": "call-1"}]
        messages = [
            AIMessage(content="calling tool", tool_calls=tool_calls),
            ToolMessage(content='{"result": 1}', tool_call_id="call-1"),
        ]
        return StudentStepResult(trace="trace", output=output, messages=messages, attempts=self._attempts)

    async def asynthesize_final_answer(self, task: str, step_summaries):
        return "done"


class FakeTeacher:
    async def areview_plan(self, task: str, plan: Plan) -> Plan:
        return plan

    async def avalidate_step(self, step: Step, trace: str, output: str):
        return {"valid": output == "success", "rationale": ""}

    async def agenerate_guidance(self, step: Step, evaluation):
        return "retry"

    def collect_results(self, items):
        return items


class FakeEvaluator:
    async def ajudge(self, context: JudgeContext):
        score = 0.9 if context.attempt >= 2 else 0.2
        return {"score": score, "judges": []}


def build_orchestrator():
    orchestration_config = OrchestrationConfig(max_retries=1, step_timeout_seconds=900, rim_guidance_tag="tag", emit_intermediate_steps=True)
    rim_config = RIMConfig(
        judges=[
            JudgeConfig(identifier="process", kind=JudgeKind.PROCESS, llm=LLMParameters(model="stub"))
        ],
        temperatures=[0.0],
        variance_threshold=1.0,
        uncertainty_threshold=1.0,
        arbiter=LLMParameters(model="arbiter"),
        success_threshold=0.7,
        retry_threshold=0.6,
        aggregation_strategy="weighted_mean",
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
    step_result = result.step_results[0]
    assert step_result.attempts == 2
    step_meta = context.metadata["steps"][1]
    event_types = [event.event_type for event in events]
    assert IntermediateStepType.TASK_START in event_types
    assert IntermediateStepType.TOOL_START in event_types
    assert IntermediateStepType.TOOL_END in event_types
    assert len(step_meta["attempts"]) == 2
    assert step_meta["guidance"] == ["retry"]
