import asyncio

import pytest

pytest.importorskip("langchain_core")

from atlas.config.models import AdapterConfig, AdapterType, LLMParameters, OrchestrationConfig, RIMConfig, StudentConfig, StudentPrompts, TeacherConfig
from atlas.transition.rewriter import RewrittenStudentPrompts, RewrittenTeacherPrompts
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
            plan=Plan(steps=[]),
            step_results=[StepResult(step_id=1, trace="", output="", evaluation={}, attempts=1)],
        )

    async def arun(self, task: str) -> Result:
        return self.result


def test_core_run_assembles_pipeline(monkeypatch):
    async def runner() -> None:
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
        monkeypatch.setattr(core, "load_config", lambda path: atlas_config)
        monkeypatch.setattr(core, "create_from_atlas_config", lambda config: StubAdapter())
        monkeypatch.setattr(core, "Student", StubStudent)
        monkeypatch.setattr(core, "Teacher", StubTeacher)
        monkeypatch.setattr(core, "Evaluator", StubEvaluator)
        monkeypatch.setattr(core, "Orchestrator", lambda *args, **kwargs: StubOrchestrator())

        class StubRewriteEngine:
            def __init__(self, *_args, **_kwargs):
                pass

            async def generate(self, *args, **kwargs):
                return (
                    RewrittenStudentPrompts("planner", "executor", "synth"),
                    RewrittenTeacherPrompts("plan", "validate", "guide"),
                )

        monkeypatch.setattr(core, "PromptRewriteEngine", StubRewriteEngine)
        result = await core.arun("task", "config.yaml")
        assert result.final_answer == "answer"

    asyncio.run(runner())
