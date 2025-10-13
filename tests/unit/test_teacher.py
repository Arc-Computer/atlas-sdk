import asyncio
import json

import pytest

from atlas.config.models import LLMParameters, LLMProvider, TeacherConfig
from atlas.personas.teacher import Teacher
from atlas.prompts import RewrittenTeacherPrompts
from atlas.types import Plan, Step
from atlas.runtime.orchestration.execution_context import ExecutionContext
from atlas.utils.llm_client import LLMResponse


def _gpt5_params() -> LLMParameters:
    return LLMParameters(
        provider=LLMProvider.OPENAI,
        model="gpt-5",
        temperature=1.0,
        timeout_seconds=3600.0,
        additional_headers={"OpenAI-Beta": "reasoning=1"},
        reasoning_effort="medium",
    )


def _attach_reasoning_capture(teacher: Teacher) -> dict[str, object]:
    captured: dict[str, object] = {}
    original = teacher._client.acomplete

    async def traced(messages, response_format=None, overrides=None):
        response = await original(messages, response_format, overrides)
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
            validation="You validate execution traces and reply with JSON {\"valid\": bool, \"guidance\": str | null}.",
            guidance="You provide concise corrective guidance.",
        )
        teacher = Teacher(config, prompts)
        teacher._client = _FakeTeacherClient()
        base_plan = Plan(steps=[Step(id=1, description="draft summary", depends_on=[])])
        reviewed = await teacher.areview_plan("Summarize Atlas SDK", base_plan)
        assert isinstance(reviewed, Plan)
        assert reviewed.steps
        step = Step(id=1, description="draft summary", depends_on=[])
        structured_output = {
            "status": "ok",
            "result": {"deliverable": "draft summary", "artifacts": {"tokens": 128}},
            "deliverable": "draft summary",
            "artifacts": {"tokens": 128},
            "text": "draft summary ready",
        }
        validation = await teacher.avalidate_step(
            step,
            "trace log",
            structured_output,
            prior_results={},
            prior_guidance=[],
            attempt_guidance=[],
        )
        assert set(validation.keys()) >= {"valid", "guidance"}
        assert isinstance(validation["valid"], bool)
        assert validation["guidance"] is None or isinstance(validation["guidance"], str)
        guidance = await teacher.agenerate_guidance(step, {"validation": {"guidance": "Take another pass."}})
        assert guidance == "Take another pass."

    asyncio.run(runner())


class _FakeTeacherClient:
    def __init__(self) -> None:
        self.reasoning = {"reasoning_content": [{"type": "thought", "text": "check constraints"}]}

    async def acomplete(self, messages, response_format=None, overrides=None):
        if response_format and response_format.get("type") == "json_object":
            user_payload = messages[-1]["content"]
            if "steps" in user_payload:
                payload = {"steps": [{"id": 1, "description": "draft summary", "depends_on": [], "tool": None, "tool_params": None}]}
            else:
                payload = {"valid": True, "guidance": "looks good"}
            return LLMResponse(content=json.dumps(payload), raw={}, reasoning=self.reasoning)
        return LLMResponse(content="Provide more detail.", raw={}, reasoning=self.reasoning)


def test_teacher_records_reasoning_metadata():
    async def runner() -> None:
        ExecutionContext.get().reset()
        config = TeacherConfig(
            llm=_gpt5_params(),
            max_review_tokens=1024,
            plan_cache_seconds=0,
            guidance_max_tokens=512,
            validation_max_tokens=512,
        )
        prompts = RewrittenTeacherPrompts(
            plan_review="Respond with JSON containing a 'steps' array.",
            validation="Return JSON {\"valid\": bool, \"guidance\": str | null}.",
            guidance="Return short guidance.",
        )
        teacher = Teacher(config, prompts)
        teacher._client = _FakeTeacherClient()
        plan = Plan(steps=[Step(id=1, description="draft summary", depends_on=[])])
        reviewed = await teacher.areview_plan("Summarize findings", plan)
        assert reviewed.steps[0].description == "draft summary"
        structured_output = {
            "status": "ok",
            "result": {"deliverable": "draft summary", "artifacts": {"tokens": 12}},
            "deliverable": "draft summary",
            "artifacts": {"tokens": 12},
            "text": "draft summary ready",
        }
        validation = await teacher.avalidate_step(
            plan.steps[0],
            "trace",
            structured_output,
            prior_results={},
            prior_guidance=[],
            attempt_guidance=[],
        )
        assert validation["reasoning"]["reasoning_content"][0]["text"] == "check constraints"
        guidance = await teacher.agenerate_guidance(
            plan.steps[0],
            {"validation": {"guidance": "Provide more detail."}},
        )
        assert guidance == "Provide more detail."
        store = ExecutionContext.get().metadata.get("reasoning_traces", {})
        assert "teacher" in store and store["teacher"], "Teacher reasoning should be recorded"
    asyncio.run(runner())


class _StaticResponseClient:
    def __init__(self, payload: dict[str, object], *, reasoning: dict[str, object] | None = None) -> None:
        self._payload = payload
        self.reasoning = reasoning or {}

    async def acomplete(self, messages, response_format=None, overrides=None):
        return LLMResponse(content=json.dumps(self._payload), raw={}, reasoning=self.reasoning)


def test_adaptive_probe_prefers_auto_mode_with_helpful_stats():
    async def runner() -> None:
        ExecutionContext.get().reset()
        config = TeacherConfig(
            llm=_gpt5_params(),
            max_review_tokens=1024,
            plan_cache_seconds=0,
            guidance_max_tokens=512,
            validation_max_tokens=512,
        )
        prompts = RewrittenTeacherPrompts(
            plan_review="Respond with JSON containing a 'steps' array.",
            validation="Return JSON {\"valid\": bool, \"guidance\": str | null}.",
            guidance="Return short guidance.",
        )
        teacher = Teacher(config, prompts)
        metadata = {
            "persona_memories": {
                "teacher_validation": [
                    {
                        "metadata": {
                            "helpful_count": 3,
                            "harmful_count": 0,
                            "neutral_count": 1,
                            "tags": ["persona:senior", "domain:finops"],
                        }
                    }
                ]
            }
        }
        dossier = {"summary": "known workflow", "risks": [{"severity": "medium"}]}
        decision = await teacher.adaptive_probe(
            "refine report",
            dossier,
            fingerprint="fp-1",
            execution_metadata=metadata,
        )
        assert decision["mode"] == "auto"
        assert decision["confidence"] == pytest.approx(0.9, rel=1e-2)
        assert "persona_helpful_ratio=0.75" in decision["evidence"]
        assert decision["recommended_personas"] == ["senior"]

    asyncio.run(runner())


def test_adaptive_probe_escalates_on_high_risk_or_negative_signals():
    async def runner() -> None:
        ExecutionContext.get().reset()
        config = TeacherConfig(
            llm=_gpt5_params(),
            max_review_tokens=1024,
            plan_cache_seconds=0,
            guidance_max_tokens=512,
            validation_max_tokens=512,
        )
        prompts = RewrittenTeacherPrompts(
            plan_review="Respond with JSON containing a 'steps' array.",
            validation="Return JSON {\"valid\": bool, \"guidance\": str | null}.",
            guidance="Return short guidance.",
        )
        teacher = Teacher(config, prompts)
        metadata = {
            "persona_memories": {
                "teacher_guidance": [
                    {
                        "metadata": {
                            "helpful_count": 1,
                            "harmful_count": 2,
                            "neutral_count": 0,
                            "tags": ["persona:analyst"],
                        }
                    }
                ]
            }
        }
        dossier = {"summary": "critical outage", "risks": [{"severity": "critical"}]}
        decision = await teacher.adaptive_probe(
            "restore service",
            dossier,
            fingerprint="fp-2",
            execution_metadata=metadata,
        )
        assert decision["mode"] == "escalate"
        assert decision["confidence"] == pytest.approx(0.35, rel=1e-2)
        assert "risk_high_severity" in decision["evidence"]
        assert "persona_helpful_ratio=0.33" in decision["evidence"]

    asyncio.run(runner())


def test_avalidate_step_injects_certification_reward_in_paired_mode():
    async def runner() -> None:
        ExecutionContext.get().reset()
        ExecutionContext.get().metadata["adaptive"] = {"active_mode": "paired", "certification_run": True}
        config = TeacherConfig(
            llm=_gpt5_params(),
            max_review_tokens=1024,
            plan_cache_seconds=0,
            guidance_max_tokens=512,
            validation_max_tokens=512,
        )
        prompts = RewrittenTeacherPrompts(
            plan_review="Respond with JSON containing a 'steps' array.",
            validation="Return JSON {\"valid\": bool, \"guidance\": str | null}.",
            guidance="Return JSON guidance.",
        )
        teacher = Teacher(config, prompts)
        teacher._client = _StaticResponseClient({"valid": True, "guidance": "All clear."})
        step = Step(id=1, description="compile", depends_on=[])
        structured_output = {
            "status": "ok",
            "deliverable": {"report": "complete"},
            "reason": "Meets certification bar.",
        }
        result = await teacher.avalidate_step(
            step,
            trace="trace",
            structured_output=structured_output,
            prior_results={},
            prior_guidance=[],
            attempt_guidance=[],
        )
        assert result["valid"] is True
        assert result["guidance"] == "All clear."
        reward = result.get("certification_reward")
        assert isinstance(reward, dict), "Certification reward should be attached in paired certification runs"
        assert reward["score"] == pytest.approx(0.95, rel=1e-3)
        assert reward["raw"]["structured_output"]["reason"] == "Meets certification bar."

    import pytest

    asyncio.run(runner())


def test_avalidate_step_trims_guidance_in_coach_mode():
    async def runner() -> None:
        ExecutionContext.get().reset()
        ExecutionContext.get().metadata["adaptive"] = {"active_mode": "coach"}
        config = TeacherConfig(
            llm=_gpt5_params(),
            max_review_tokens=1024,
            plan_cache_seconds=0,
            guidance_max_tokens=512,
            validation_max_tokens=512,
        )
        prompts = RewrittenTeacherPrompts(
            plan_review="Respond with JSON containing a 'steps' array.",
            validation="Return JSON {\"valid\": bool, \"guidance\": str | null}.",
            guidance="Return JSON guidance.",
        )
        teacher = Teacher(config, prompts)
        guidance_text = "First sentence. Second sentence. Third sentence with extra detail."
        teacher._client = _StaticResponseClient({"valid": False, "guidance": guidance_text})
        step = Step(id=1, description="draft", depends_on=[])
        structured_output = {
            "status": "needs_support",
        }
        result = await teacher.avalidate_step(
            step,
            trace="trace",
            structured_output=structured_output,
            prior_results={},
            prior_guidance=[],
            attempt_guidance=[],
        )
        assert result["valid"] is False
        assert result["guidance"] == "First sentence. Second sentence"

    asyncio.run(runner())
