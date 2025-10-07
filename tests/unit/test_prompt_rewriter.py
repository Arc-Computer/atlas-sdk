import asyncio
import json

from atlas.config.models import (
    AdapterConfig,
    AdapterType,
    LLMParameters,
    LLMProvider,
    PromptRewriteConfig,
    StudentConfig,
    TeacherConfig,
)
from atlas.transition.rewriter import PromptRewriteEngine
from atlas.transition.rewriter import RewrittenStudentPrompts, RewrittenTeacherPrompts
from atlas.utils.llm_client import LLMResponse


def _gpt5_params() -> LLMParameters:
    return LLMParameters(
        provider=LLMProvider.OPENAI,
        model="gpt-5",
        temperature=1.0,
        timeout_seconds=3600.0,
        additional_headers={"OpenAI-Beta": "reasoning=1"},
    )


def test_prompt_rewrite_engine_handles_persona_parallelism():
    async def runner() -> None:
        engine = PromptRewriteEngine(
            PromptRewriteConfig(llm=_gpt5_params(), max_tokens=4096, temperature=1.0),
            fallback_llm=None,
        )
        responses = {
            "planner": "Design sequential steps that expose explicit dependencies and clear parallelisable stages.",
            "executor": "Execute the assigned step, referencing context outputs and remaining safe for parallel peers.",
            "synthesizer": "Combine every completed step outcome with citations and highlight residual gaps.",
            "plan_review": "Check dependency integrity, coverage, and parallel readiness before approving the plan.",
            "validation": "Validate each attempt with structured reasoning and expected outputs while noting concurrency issues.",
            "guidance": "Deliver actionable guidance tailored to the failed attempt so the retry succeeds quickly.",
        }

        async def fake_acomplete(messages, response_format=None, overrides=None):
            payload = json.loads(messages[1]["content"])
            persona_key = payload["persona"]["output_key"]
            text = responses[persona_key]
            return LLMResponse(content=text, raw={"persona": persona_key})

        engine._client.acomplete = fake_acomplete
        adapter_config = AdapterConfig(type=AdapterType.HTTP, name="agent", system_prompt="Base prompt", tools=[])
        student = StudentConfig()
        teacher = TeacherConfig(
            llm=_gpt5_params(),
            max_review_tokens=3072,
            plan_cache_seconds=0,
            guidance_max_tokens=1024,
            validation_max_tokens=1024,
        )
        student_prompts, teacher_prompts = await engine.generate(
            base_prompt="Base prompt",
            adapter_config=adapter_config,
            student_config=student,
            teacher_config=teacher,
        )
        assert isinstance(student_prompts, RewrittenStudentPrompts)
        assert isinstance(teacher_prompts, RewrittenTeacherPrompts)
        assert student_prompts.planner.startswith("Base prompt")
        assert teacher_prompts.guidance.startswith("Base prompt")
        assert responses["guidance"] in teacher_prompts.guidance

    asyncio.run(runner())
