import asyncio

import pytest

from atlas.config.models import LLMParameters, LLMProvider, TeacherConfig
from atlas.roles.teacher import Teacher
from atlas.transition.rewriter import RewrittenTeacherPrompts
from atlas.types import Plan, Step


def _gpt5_params() -> LLMParameters:
    return LLMParameters(
        provider=LLMProvider.OPENAI,
        model="gpt-5",
        temperature=1.0,
        timeout_seconds=3600.0,
         additional_headers={"OpenAI-Beta": "reasoning=1"},
    )


def _attach_reasoning_capture(teacher: Teacher) -> dict[str, object]:
    captured: dict[str, object] = {}
    original = teacher._client.acomplete

    async def traced(messages, response_format=None, overrides=None):
        merged = dict(overrides or {})
        extra_body = dict(merged.get("extra_body") or {})
        extra_body.setdefault("reasoning_effort", "medium")
        merged["extra_body"] = extra_body
        response = await original(messages, response_format, merged)
        captured["content"] = response.content
        captured["raw"] = response.raw
        return response

    teacher._client.acomplete = traced
    return captured


def test_teacher_live_contracts():
    async def runner() -> None:
        config = TeacherConfig(
            llm=_gpt5_params(),
            max_review_tokens=3072,
            plan_cache_seconds=0,
            guidance_max_tokens=1536,
            validation_max_tokens=1536,
        )
        prompts = RewrittenTeacherPrompts(
            plan_review="You review plans and respond with JSON containing a 'steps' array.",
            validation="You validate execution traces and reply with JSON {\"valid\": bool, \"rationale\": str}.",
            guidance="You provide concise corrective guidance.",
        )
        teacher = Teacher(config, prompts)
        captured = _attach_reasoning_capture(teacher)
        base_plan = Plan(steps=[Step(id=1, description="draft summary", depends_on=[])])
        try:
            reviewed = await teacher.areview_plan("Summarize Atlas SDK", base_plan)
        except Exception:
            raw = captured.get("content", "")
            if raw:
                print(f"Teacher plan raw response: {raw}")
            else:
                print(f"Teacher plan payload: {captured.get('raw', {})}")
            raise
        assert isinstance(reviewed, Plan)
        assert reviewed.steps
        step = Step(id=1, description="draft summary", depends_on=[])
        try:
            validation = await teacher.avalidate_step(step, "trace log", "output content")
        except Exception:
            raw = captured.get("content", "")
            if raw:
                print(f"Teacher validation raw response: {raw}")
            else:
                print(f"Teacher validation payload: {captured.get('raw', {})}")
            raise
        assert set(validation.keys()) >= {"valid", "rationale"}
        assert isinstance(validation["valid"], bool)
        assert validation["rationale"].strip()
        try:
            guidance = await teacher.agenerate_guidance(step, {"score": 0.5})
        except Exception:
            raw = captured.get("content", "")
            if raw:
                print(f"Teacher guidance raw response: {raw}")
            else:
                print(f"Teacher guidance payload: {captured.get('raw', {})}")
            raise
        assert guidance.strip()

    asyncio.run(runner())
