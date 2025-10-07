import io

from atlas.data_models.intermediate_step import IntermediateStepPayload
from atlas.data_models.intermediate_step import IntermediateStepType
from atlas.data_models.intermediate_step import StreamEventData
from atlas.orchestration.execution_context import ExecutionContext
from atlas.telemetry import ConsoleTelemetryStreamer
from atlas.runtime.schema import AtlasRewardBreakdown
from atlas.types import Plan
from atlas.types import Result
from atlas.types import Step
from atlas.types import StepEvaluation
from atlas.types import StepResult


def test_console_streamer_renders_events_and_summary():
    context = ExecutionContext.get()
    context.reset()
    context.metadata["plan"] = {"steps": [{"id": 1, "description": "collect data"}]}
    stream = io.StringIO()
    streamer = ConsoleTelemetryStreamer(output=stream)
    streamer.attach(context)
    streamer.session_started("demo task")

    manager = context.intermediate_step_manager
    manager.push_intermediate_step(
        IntermediateStepPayload(
            UUID="step-1",
            event_type=IntermediateStepType.TASK_START,
            name="step_1",
            data=StreamEventData(
                input={
                    "step": {"id": 1, "description": "collect data"},
                    "guidance": ["Focus on accuracy"],
                    "attempt": 2,
                }
            ),
        )
    )
    manager.push_intermediate_step(
        IntermediateStepPayload(
            UUID="tool-1",
            event_type=IntermediateStepType.TOOL_START,
            name="search",
            data=StreamEventData(input={"query": "dataset"}),
        )
    )
    manager.push_intermediate_step(
        IntermediateStepPayload(
            UUID="tool-1",
            event_type=IntermediateStepType.TOOL_END,
            name="search",
            data=StreamEventData(output={"result": "found"}),
        )
    )
    manager.push_intermediate_step(
        IntermediateStepPayload(
            UUID="step-1",
            event_type=IntermediateStepType.TASK_END,
            name="step_1",
            data=StreamEventData(
                output={
                    "trace": "trace",
                    "output": "complete",
                    "evaluation": {
                        "validation": {"valid": True, "rationale": "looks good"},
                        "reward": {"score": 0.8, "uncertainty": 0.1, "judges": [1]},
                    },
                }
            ),
        )
    )

    result = Result(
        final_answer="done",
        plan=Plan(steps=[Step(id=1, description="collect data")]),
        step_results=[
            StepResult(
                step_id=1,
                trace="trace",
                output="complete",
                evaluation=StepEvaluation(
                    validation={},
                    reward=AtlasRewardBreakdown(score=0.8, raw={"score": 0.8}),
                ),
                attempts=2,
            )
        ],
    )
    streamer.session_completed(result)
    streamer.detach()

    output = stream.getvalue()
    assert "Atlas task started" in output
    assert "Plan ready" in output
    assert "retry 2" in output
    assert "guidance" in output
    assert "reward score: 0.80" in output
    assert "RIM scores" in output


def test_console_streamer_failure_message():
    context = ExecutionContext.get()
    context.reset()
    stream = io.StringIO()
    streamer = ConsoleTelemetryStreamer(output=stream)
    streamer.attach(context)
    streamer.session_started("demo task")
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        streamer.session_failed(exc)
    streamer.detach()
    output = stream.getvalue()
    assert "failed" in output
    assert "boom" in output
