import asyncio
import json

import pytest

pytest.importorskip("langchain_core")

from atlas.agent.registry import AgentAdapter
from atlas.orchestration.execution_context import ExecutionContext
from atlas.config.models import AdapterType
from atlas.config.models import HTTPAdapterTransport
from atlas.config.models import StudentConfig
from atlas.config.models import StudentPrompts
from atlas.config.models import ToolDefinition
from atlas.config.models import ToolParameterSchema
from atlas.config.models import AdapterConfig
from atlas.config.models import RetryPolicy
from atlas.roles.student import Student
from atlas.transition.rewriter import RewrittenStudentPrompts
from atlas.types import Plan, Step


class StubAdapter(AgentAdapter):
    def __init__(self):
        self.plan_response = {
            "steps": [
                {"id": 1, "description": "step one", "depends_on": [], "tool": None, "tool_params": None, "estimated_time": "1m"}
            ],
            "total_estimated_time": "1m",
        }

    async def ainvoke(self, prompt: str, metadata: dict | None = None) -> str:
        mode = (metadata or {}).get("mode")
        if mode == "planning":
            return json.dumps(self.plan_response)
        if mode == "synthesis":
            return "Final answer"
        return "{\"content\": \"execution complete\"}"


def student_config() -> StudentConfig:
    return StudentConfig(
        prompts=StudentPrompts(
            planner="Planner {base_prompt}",
            executor="Executor {base_prompt}",
            synthesizer="Synth {base_prompt}",
        )
    )


def adapter_config() -> AdapterConfig:
    tool_parameters = ToolParameterSchema(properties={}, required=[], additionalProperties=False)
    tool_def = ToolDefinition(name="noop", description="No operation", parameters=tool_parameters)
    return AdapterConfig(
        type=AdapterType.HTTP,
        name="stub",
        system_prompt="Base prompt",
        tools=[tool_def],
    )


@pytest.mark.asyncio
async def test_student_plan_execute_and_synthesize():
    ExecutionContext.get().reset()
    adapter = StubAdapter()
    adapter_cfg = adapter_config()
    config = student_config()
    rewrites = RewrittenStudentPrompts(
        planner="planner",
        executor="executor",
        synthesizer="synth",
    )
    student = Student(
        adapter=adapter,
        adapter_config=adapter_cfg,
        student_config=config,
        student_prompts=rewrites,
    )
    plan = await student.acreate_plan("Do something")
    assert isinstance(plan, Plan)
    step = plan.steps[0]
    result = await student.aexecute_step(step, context={}, guidance=[])
    assert "execution" in result.output.lower()
    final_answer = await student.asynthesize_final_answer("Do something", [])
    assert final_answer == "Final answer"
