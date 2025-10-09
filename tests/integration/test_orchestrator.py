import pytest
from typing import Any

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
        self.observed_context: dict[int, Any] | None = None

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
        metadata: dict[str, Any] = {"attempt": attempts, "step_id": step.id}
        if step.id == 1:
            output = "success-a" if attempts >= 2 else "pending-a"
            tool_id = "call-1"
            if output.startswith("success"):
                metadata["structured_data"] = {"dataset": "A", "rows": [1, 2, 3]}
        elif step.id == 2:
            output = "success-b"
            tool_id = "call-2"
            metadata["structured_output"] = {"dataset": "B", "rows": [4, 5]}
        else:
            entry_a = context.get(1, {})
            entry_b = context.get(2, {})
            if isinstance(entry_a, dict):
                part_a = entry_a.get("output_text", "")
            else:
                part_a = entry_a
            if isinstance(entry_b, dict):
                part_b = entry_b.get("output_text", "")
            else:
                part_b = entry_b
            cache_a = entry_a.get("cached_data", {}) if isinstance(entry_a, dict) else {}
            cache_b = entry_b.get("cached_data", {}) if isinstance(entry_b, dict) else {}
            metadata["context_snapshot"] = {1: entry_a, 2: entry_b}
            self.observed_context = metadata["context_snapshot"]
            output = f"summary:{part_a}|{part_b}"
            if cache_a or cache_b:
                metadata["cached_data"] = {"upstream": {"a": cache_a, "b": cache_b}}
            tool_id = "call-3"
        tool_calls = [{"name": "search", "args": {"query": step.description}, "id": tool_id}]
        messages = [
            AIMessage(content="calling tool", tool_calls=tool_calls),
            ToolMessage(content='{"result": 1}', tool_call_id=tool_id),
        ]
        return StudentStepResult(trace="trace", output=output, messages=messages, attempts=attempts, metadata=metadata)

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
    def __init__(self):
        self.invocations: list[JudgeContext] = []

    async def ajudge(self, context: JudgeContext):
        self.invocations.append(context)
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
    teacher = FakeTeacher()
    student = FakeStudent()
    evaluator = FakeEvaluator()
    orchestrator = Orchestrator(
        teacher=teacher,
        student=student,
        evaluator=evaluator,
        orchestration_config=orchestration_config,
        rim_config=rim_config,
    )
    return orchestrator, teacher, student, evaluator


@pytest.mark.asyncio
async def test_orchestrator_retries_and_records_context():
    context = ExecutionContext.get()
    context.reset()
    events = []
    subscription = context.event_stream.subscribe(events.append)
    orchestrator, _, student, evaluator = build_orchestrator()
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
    attempt_1 = step_meta["attempts"][0]
    assert attempt_1["reward_skipped"] is True
    assert "timings_ms" in attempt_1 and attempt_1["timings_ms"]["validation_ms"] >= 0.0
    # Only validated outputs should appear in downstream context
    assert student.observed_context is not None
    assert student.observed_context[1]["output_text"] == "success-a"
    assert "cached_data" in student.observed_context[1]
    assert student.observed_context[2]["cached_data"]["dataset"] == "B"
    # Judges should only run on successful attempts
    assert len(evaluator.invocations) == 3
    assert all(inv.output.startswith(("success", "summary")) for inv in evaluator.invocations)
    # Structured data should persist into step metadata and results
    results_by_step = {entry.step_id: entry for entry in result.step_results}
    assert results_by_step[1].metadata["cached_data"]["dataset"] == "A"
    assert results_by_step[3].metadata["runtime"]["reward_skipped"] is False
