import asyncio
import json

import pytest

from atlas.config.models import (
    AdapterConfig,
    AdapterType,
    LLMParameters,
    LLMProvider,
    PromptRewriteConfig,
    StudentConfig,
    StudentPrompts,
    TeacherConfig,
)
from atlas.transition.rewriter import PromptRewriteEngine
from atlas.transition.rewriter import RewrittenStudentPrompts, RewrittenTeacherPrompts


def _gpt5_params() -> LLMParameters:
    return LLMParameters(
        provider=LLMProvider.OPENAI,
        model="gpt-5",
        temperature=1.0,
        timeout_seconds=3600.0,
        additional_headers={"OpenAI-Beta": "reasoning=1"},
    )


def test_prompt_rewrite_engine_parses_live_response():
    async def runner() -> None:
        engine = PromptRewriteEngine(
            PromptRewriteConfig(llm=_gpt5_params(), max_tokens=4096, temperature=1.0),
            fallback_llm=None,
        )
        captured: dict[str, object] = {}
        original_acomplete = engine._client.acomplete

        async def traced_acomplete(messages, response_format=None, overrides=None):
            merged_overrides = dict(overrides or {})
            extra_body = dict(merged_overrides.get("extra_body") or {})
            extra_body.setdefault("reasoning_effort", "medium")
            merged_overrides["extra_body"] = extra_body
            response = await original_acomplete(messages, response_format, merged_overrides)
            captured["content"] = response.content
            captured["raw"] = response.raw
            return response

        engine._client.acomplete = traced_acomplete
        adapter_config = AdapterConfig(type=AdapterType.HTTP, name="agent", system_prompt="Base prompt", tools=[])
        student = StudentConfig(prompts=StudentPrompts(planner="", executor="", synthesizer=""))
        teacher = TeacherConfig(
            llm=_gpt5_params(),
            max_review_tokens=3072,
            plan_cache_seconds=0,
            guidance_max_tokens=1024,
            validation_max_tokens=1024,
        )
        try:
            student_prompts, teacher_prompts = await engine.generate(
                base_prompt="Base prompt",
                adapter_config=adapter_config,
                student_config=student,
                teacher_config=teacher,
            )
        except Exception:
            raw = captured.get("content", "")
            if raw:
                print(f"Prompt rewrite raw response: {raw}")
            else:
                print(f"Prompt rewrite payload: {captured.get('raw', {})}")
            raise
        raw_json = captured.get("content", "")
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError:
            print(f"Prompt rewrite raw response: {raw_json}")
            raise
        assert isinstance(student_prompts, RewrittenStudentPrompts)
        assert isinstance(teacher_prompts, RewrittenTeacherPrompts)
        assert payload["student"]["planner"].strip()
        assert payload["teacher"]["guidance"].strip()

    asyncio.run(runner())
