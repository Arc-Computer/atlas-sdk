from atlas.config.models import (
    LLMParameters,
    LLMProvider,
    StudentConfig,
    StudentPrompts,
    TeacherConfig,
)
from atlas.prompts import (
    RewrittenStudentPrompts,
    RewrittenTeacherPrompts,
    build_student_prompts,
    build_teacher_prompts,
)


def _gpt5_params() -> LLMParameters:
    return LLMParameters(
        provider=LLMProvider.OPENAI,
        model="gpt-5",
        temperature=1.0,
        timeout_seconds=3600.0,
        additional_headers={"OpenAI-Beta": "reasoning=1"},
    )


def test_build_student_prompts_respects_templates():
    student_cfg = StudentConfig(
        prompts=StudentPrompts(
            planner="{base_prompt}\nPlan the work carefully.",
            executor="{base_prompt}\nRun the assigned step and emit JSON.",
            synthesizer="Summarize outputs with citations.",
        )
    )
    prompts = build_student_prompts("Base prompt", student_cfg)
    assert isinstance(prompts, RewrittenStudentPrompts)
    assert prompts.planner.startswith("Base prompt")
    assert prompts.executor.startswith("Base prompt")
    assert prompts.synthesizer == "Summarize outputs with citations."


def test_build_teacher_prompts_default_templates_include_base_prompt():
    teacher_cfg = TeacherConfig(
        llm=_gpt5_params(),
        max_review_tokens=2048,
        plan_cache_seconds=0,
        guidance_max_tokens=512,
        validation_max_tokens=512,
    )
    prompts = build_teacher_prompts("Base prompt", teacher_cfg)
    assert isinstance(prompts, RewrittenTeacherPrompts)
    assert prompts.plan_review.startswith("Base prompt")
    assert "Return JSON only" in prompts.plan_review
    assert prompts.validation.startswith("Base prompt")
    assert prompts.guidance.startswith("Base prompt")
    assert "Return JSON only" in prompts.validation
