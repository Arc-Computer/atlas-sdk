import asyncio

import pytest

from atlas.config.models import (
    AdapterConfig,
    AdapterType,
    LLMParameters,
    OrchestrationConfig,
    RIMConfig,
    StudentConfig,
    StudentPrompts,
    TeacherConfig,
)
from atlas.prompts import RewrittenStudentPrompts, RewrittenTeacherPrompts
from atlas.runtime.orchestration.execution_context import ExecutionContext
from atlas.runtime.schema import AtlasRewardBreakdown
from atlas.types import Plan, Result, StepEvaluation, StepResult


class StubAdapter:
    async def ainvoke(self, prompt: str, metadata=None):
        return "{}"


class StubStudent:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def update_prompts(self, student_prompts: RewrittenStudentPrompts) -> None:
        pass


class StubTeacher:
    def __init__(self, *args, **kwargs) -> None:
        pass

    def update_prompts(self, prompts: RewrittenTeacherPrompts) -> None:
        pass


class StubEvaluator:
    def __init__(self, *args, **kwargs) -> None:
        pass


class ModeSettingOrchestrator:
    def __init__(self, mode: str, *args, **kwargs) -> None:
        self._mode = mode
        self._persona_refresh = kwargs.get("persona_refresh")
        reward = AtlasRewardBreakdown(score=1.0)
        evaluation = StepEvaluation(validation={}, reward=reward)
        self.result = Result(
            final_answer=f"{mode}-run",
            plan=Plan(steps=[], execution_mode=mode),
            step_results=[StepResult(step_id=1, trace="", output="", evaluation=evaluation, attempts=1)],
        )

    async def arun(self, task: str) -> Result:
        context = ExecutionContext.get()
        context.metadata.setdefault("steps", {})
        context.metadata["execution_mode"] = self._mode
        if self._persona_refresh is not None:
            await self._persona_refresh()
        return self.result


def test_fingerprint_includes_final_execution_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    from atlas import core

    adapter_config = AdapterConfig(type=AdapterType.HTTP, name="mode-agent", system_prompt="Base prompt", tools=[])
    shared_config = type(
        "Config",
        (),
        {
            "agent": adapter_config,
            "student": StudentConfig(prompts=StudentPrompts(planner="{base_prompt}", executor="{base_prompt}", synthesizer="{base_prompt}")),
            "teacher": TeacherConfig(llm=LLMParameters(model="model")),
            "orchestration": OrchestrationConfig(max_retries=0, step_timeout_seconds=10, rim_guidance_tag="tag", emit_intermediate_steps=True),
            "rim": RIMConfig(
                small_model=LLMParameters(model="stub"),
                large_model=LLMParameters(model="arbiter"),
                active_judges={"process": True},
                variance_threshold=1.0,
                uncertainty_threshold=1.0,
                parallel_workers=1,
            ),
            "storage": None,
            "prompt_rewrite": None,
        },
    )()

    def configure(mode: str) -> None:
        monkeypatch.setattr(core, "load_config", lambda path: shared_config)
        monkeypatch.setattr(core, "create_from_atlas_config", lambda config: StubAdapter())
        monkeypatch.setattr(core, "Student", StubStudent)
        monkeypatch.setattr(core, "Teacher", StubTeacher)
        monkeypatch.setattr(core, "Evaluator", StubEvaluator)
        monkeypatch.setattr(
            core,
            "Orchestrator",
            lambda *args, **kwargs: ModeSettingOrchestrator(mode, *args, **kwargs),
        )
        monkeypatch.setattr(
            core,
            "build_student_prompts",
            lambda *_args, **_kwargs: RewrittenStudentPrompts("planner", "executor", "synth"),
        )
        monkeypatch.setattr(
            core,
            "build_teacher_prompts",
            lambda *_args, **_kwargs: RewrittenTeacherPrompts("plan", "validate", "guide"),
        )

    async def _run(mode: str) -> str:
        configure(mode)
        await core.arun("fingerprint-task", "config.yaml")
        context = ExecutionContext.get()
        inputs = context.metadata.get("persona_fingerprint_inputs")
        assert inputs is not None
        assert inputs.execution_mode == mode
        fingerprint = context.metadata.get("persona_fingerprint")
        assert fingerprint
        return fingerprint

    fingerprint_stepwise = asyncio.run(_run("stepwise"))
    fingerprint_single_shot = asyncio.run(_run("single_shot"))

    assert fingerprint_stepwise != fingerprint_single_shot
