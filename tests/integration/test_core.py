import asyncio

import pytest

from atlas.config.models import AdapterConfig, AdapterType, JudgeConfig, JudgeKind, LLMParameters, OrchestrationConfig, RIMConfig, StudentConfig, StudentPrompts, TeacherConfig
from atlas.types import Plan, Result, StepResult


class StubAdapter:
    async def ainvoke(self, prompt: str, metadata=None):
        return "{}"


class StubStudent:
    def __init__(self, *args, **kwargs):
        pass


class StubTeacher:
    def __init__(self, *args, **kwargs):
        pass


class StubEvaluator:
    def __init__(self, *args, **kwargs):
        pass


class StubOrchestrator:
    def __init__(self, *args, **kwargs):
        self.result = Result(
            final_answer="answer",
            plan=Plan(steps=[], total_estimated_time="0m"),
            step_results=[StepResult(step_id=1, trace="", output="", evaluation={}, attempts=1)],
        )

    async def arun(self, task: str) -> Result:
        return self.result


@pytest.mark.asyncio
async def test_core_run_assembles_pipeline(monkeypatch):
    from atlas import core

    atlas_config = type(
        "Config",
        (),
        {
            "agent": AdapterConfig(type=AdapterType.HTTP, name="stub", system_prompt="Base", tools=[]),
            "student": StudentConfig(prompts=StudentPrompts(planner="{base_prompt}", executor="{base_prompt}", synthesizer="{base_prompt}")),
            "teacher": TeacherConfig(llm=LLMParameters(model="model")),
            "orchestration": OrchestrationConfig(max_retries=0, step_timeout_seconds=10, rim_guidance_tag="tag", emit_intermediate_steps=True),
            "rim": RIMConfig(
                judges=[JudgeConfig(identifier="process", kind=JudgeKind.PROCESS, llm=LLMParameters(model="stub"))],
                temperatures=[0.0],
                variance_threshold=1.0,
                uncertainty_threshold=1.0,
                arbiter=LLMParameters(model="arbiter"),
                success_threshold=0.7,
                retry_threshold=0.6,
                aggregation_strategy="weighted_mean",
            ),
            "storage": None,
        },
    )()
    monkeypatch.setattr(core, "load_config", lambda path: atlas_config)
    monkeypatch.setattr(core, "create_from_atlas_config", lambda config: StubAdapter())
    monkeypatch.setattr(core, "Student", StubStudent)
    monkeypatch.setattr(core, "Teacher", StubTeacher)
    monkeypatch.setattr(core, "Evaluator", StubEvaluator)
    monkeypatch.setattr(core, "Orchestrator", lambda *args, **kwargs: StubOrchestrator())
    result = await core.arun("task", "config.yaml")
    assert result.final_answer == "answer"
