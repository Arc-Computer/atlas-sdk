import asyncio
import json

import pytest

from atlas.config.models import (
    AdapterConfig,
    AdapterType,
    LLMParameters,
    PromptRewriteConfig,
    StudentConfig,
    StudentPrompts,
    TeacherConfig,
)
from atlas.transition.rewriter import (
    PromptRewriteEngine,
)
from atlas.transition.rewriter import RewrittenStudentPrompts, RewrittenTeacherPrompts


class StubLLMClient:
    def __init__(self, *_args, **_kwargs):
        pass

    async def acomplete(self, messages, response_format=None, overrides=None):
        return type(
            "Resp",
            (),
            {
                "content": json.dumps(
                    {
                        "student": {
                            "planner": "planner prompt",
                            "executor": "executor prompt",
                            "synthesizer": "synth prompt",
                        },
                        "teacher": {
                            "plan_review": "plan review",
                            "validation": "validation",
                            "guidance": "guidance",
                        },
                    }
                ),
                "raw": {},
            },
        )


@pytest.mark.asyncio
async def test_prompt_rewrite_engine_parses_json(monkeypatch):
    monkeypatch.setattr("atlas.transition.rewriter.LLMClient", lambda *_: StubLLMClient())
    engine = PromptRewriteEngine(
        PromptRewriteConfig(llm=LLMParameters(model="stub")),
        fallback_llm=None,
    )
    adapter_config = AdapterConfig(
        type=AdapterType.HTTP,
        name="agent",
        system_prompt="Base prompt",
        tools=[],
    )
    student = StudentConfig(prompts=StudentPrompts(planner="", executor="", synthesizer=""))
    teacher = TeacherConfig(
        llm=LLMParameters(model="teacher"),
        max_review_tokens=128,
        plan_cache_seconds=0,
        guidance_max_tokens=64,
        validation_max_tokens=64,
    )
    student_prompts, teacher_prompts = await engine.generate(
        base_prompt="Base prompt",
        adapter_config=adapter_config,
        student_config=student,
        teacher_config=teacher,
    )
    assert isinstance(student_prompts, RewrittenStudentPrompts)
    assert student_prompts.planner == "planner prompt"
    assert isinstance(teacher_prompts, RewrittenTeacherPrompts)
    assert teacher_prompts.guidance == "guidance"

