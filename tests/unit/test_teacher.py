import json

import pytest

from atlas.config.models import LLMParameters
from atlas.config.models import TeacherConfig
from atlas.roles.teacher import Teacher
from atlas.types import Plan, Step
from atlas.utils.llm_client import LLMResponse
from atlas.transition.rewriter import PromptRewriter


class StubLLMClient:
    def __init__(self, *_args, **_kwargs):
        self.calls: list[tuple] = []

    async def acomplete(self, messages, response_format=None):
        self.calls.append((tuple((msg["role"], msg["content"]) for msg in messages), response_format))
        if response_format:
            return LLMResponse(content=json.dumps({"steps": [], "total_estimated_time": "0m"}), raw={})
        return LLMResponse(content="Guidance text", raw={})


@pytest.mark.asyncio
async def test_teacher_plan_review_uses_cache(monkeypatch):
    config = TeacherConfig(
        llm=LLMParameters(model="test-model"),
        max_review_tokens=256,
        plan_cache_seconds=60,
        guidance_max_tokens=128,
        validation_max_tokens=128,
    )
    stub_client = StubLLMClient()
    monkeypatch.setattr("atlas.roles.teacher.LLMClient", lambda *_: stub_client)
    prompts = PromptRewriter().rewrite_teacher("Base prompt", config.prompts)
    teacher = Teacher(config, prompts)
    plan = Plan(steps=[], total_estimated_time="0m")
    reviewed_first = await teacher.areview_plan("task", plan)
    reviewed_second = await teacher.areview_plan("task", plan)
    assert reviewed_first == reviewed_second
    assert len(stub_client.calls) == 1


@pytest.mark.asyncio
async def test_teacher_validation_and_guidance(monkeypatch):
    config = TeacherConfig(
        llm=LLMParameters(model="test-model"),
        max_review_tokens=256,
        plan_cache_seconds=0,
        guidance_max_tokens=128,
        validation_max_tokens=128,
    )
    stub_client = StubLLMClient()
    monkeypatch.setattr("atlas.roles.teacher.LLMClient", lambda *_: stub_client)
    prompts = PromptRewriter().rewrite_teacher("Base prompt", config.prompts)
    teacher = Teacher(config, prompts)
    step = Step(id=1, description="do it", depends_on=[], estimated_time="1m")
    validation = await teacher.avalidate_step(step, "trace", "output")
    assert set(validation.keys()) == {"valid", "rationale"}
    guidance = await teacher.agenerate_guidance(step, {"score": 0.2})
    assert guidance == "Guidance text"
