import asyncio
import json

import pytest

from atlas.config.models import AdaptiveProbeConfig, LLMParameters, LLMProvider, TeacherConfig
from atlas.runtime.adaptive import CapabilityProbeClient
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
        assert validation["cached"] is False
        assert teacher._client.overrides[0] and teacher._client.overrides[0].get("max_tokens") == config.max_review_tokens
        assert teacher._client.overrides[1] and teacher._client.overrides[1].get("max_tokens") == config.validation_max_tokens
        guidance = await teacher.agenerate_guidance(step, {"validation": {"guidance": "Take another pass."}})
        assert guidance == "Take another pass."

    asyncio.run(runner())


class _FakeTeacherClient:
    def __init__(self) -> None:
        self.reasoning = {"reasoning_content": [{"type": "thought", "text": "check constraints"}]}
        self.overrides: list[dict[str, object] | None] = []

    async def acomplete(self, messages, response_format=None, overrides=None):
        self.overrides.append(overrides)
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
        assert validation["cached"] is False
        assert teacher._client.overrides[0] and teacher._client.overrides[0].get("max_tokens") == config.max_review_tokens
        assert teacher._client.overrides[1] and teacher._client.overrides[1].get("max_tokens") == config.validation_max_tokens
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


def test_validation_signature_stable_and_blob_storage():
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
        teacher._client = _StaticResponseClient({"valid": True, "guidance": None})
        step = Step(id=1, description="draft summary", depends_on=[])
        structured_output = {
            "status": "ok",
            "result": {
                "deliverable": {"content": "draft summary"},
                "artifacts": {"tokens": 12},
            },
            "deliverable": {"content": "draft summary"},
            "artifacts": {"tokens": 12},
            "text": "draft summary ready",
        }
        prior_results = {0: {"status": "ok", "text": "seed"}}
        signature_one = teacher.validation_signature(step, structured_output, prior_results, [], [])
        signature_two = teacher.validation_signature(step, structured_output, prior_results, [], [])
        assert signature_one == signature_two
        mutated = dict(structured_output)
        mutated["status"] = "error"
        signature_three = teacher.validation_signature(step, mutated, prior_results, [], [])
        assert signature_three != signature_one
        await teacher.avalidate_step(
            step,
            "trace",
            structured_output,
            prior_results=prior_results,
            prior_guidance=[],
            attempt_guidance=[],
        )
        metadata = ExecutionContext.get().metadata
        blobs = metadata.get("validation_blobs")
        assert blobs and "structured_output" in blobs and blobs["structured_output"]
        assert "prior_results" in blobs and blobs["prior_results"]
        context_hashes = metadata.get("validation_context_hashes")
        assert context_hashes and context_hashes.get("prior_results")

    asyncio.run(runner())


class _PayloadCaptureClient:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []

    async def acomplete(self, messages, response_format=None, overrides=None):
        request = messages[-1]["content"]
        json_payload = request.split("\nReturn json.")[0]
        self.payloads.append(json.loads(json_payload))
        return LLMResponse(content=json.dumps({"valid": True, "guidance": None}), raw={}, reasoning={})


def test_validation_payload_rehydrates_full_content():
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
        capture = _PayloadCaptureClient()
        teacher._client = capture
        step = Step(id=1, description="draft summary", depends_on=[])
        prior_results = {0: {"status": "ok", "text": "seed"}}
        structured_output = {
            "status": "ok",
            "result": {
                "deliverable": {"content": "draft summary"},
                "artifacts": {"tokens": 12},
            },
            "deliverable": {"content": "draft summary"},
            "artifacts": {"tokens": 12},
            "text": "draft summary ready",
        }
        await teacher.avalidate_step(
            step,
            "trace",
            structured_output,
            prior_results=prior_results,
            prior_guidance=[],
            attempt_guidance=[],
        )
        assert capture.payloads, "Expected payload capture to record validation request"
        payload = capture.payloads[-1]
        assert payload["structured_output"]["content"] == teacher._jsonify(structured_output)
        assert payload["artifacts"]["content"] == teacher._jsonify(structured_output["artifacts"])
        assert payload["deliverable"]["content"] == teacher._jsonify(structured_output["deliverable"])
        assert payload["prior_results"]["content"] == teacher._jsonify(prior_results)

    asyncio.run(runner())


class _ProbeClientStub:
    def __init__(self, payload: dict[str, object], *, reasoning: dict[str, object] | None = None) -> None:
        self._payload = payload
        self.reasoning = reasoning or {}

    async def acomplete(self, messages, response_format=None, overrides=None):
        return LLMResponse(content=json.dumps(self._payload), raw={}, reasoning=self.reasoning)


def test_capability_probe_prefers_auto_mode_with_helpful_stats():
    async def runner() -> None:
        probe = CapabilityProbeClient(AdaptiveProbeConfig())
        probe._client = _ProbeClientStub(
            {
                "mode": "auto",
                "confidence": 0.92,
                "evidence": ["learning_history_average=0.88"],
            }
        )
        metadata = {"learning_history": {"entries": [], "count": 3, "average_score": 0.88}}
        dossier = {"summary": "known workflow", "risks": [{"severity": "medium"}]}
        decision = await probe.arun(
            task="refine report",
            dossier=dossier,
            execution_metadata=metadata,
        )
        assert decision.mode == "auto"
        assert decision.confidence == pytest.approx(0.92, rel=1e-2)
        assert decision.raw["evidence"] == ["learning_history_average=0.88"]

    asyncio.run(runner())


def test_capability_probe_escalates_on_high_risk_signals():
    async def runner() -> None:
        probe = CapabilityProbeClient(AdaptiveProbeConfig())
        probe._client = _ProbeClientStub(
            {
                "mode": "escalate",
                "confidence": 0.34,
                "evidence": ["learning_history_sparse", "recent_scores_below_threshold"],
            }
        )
        metadata = {"learning_history": {"entries": [], "count": 0}}
        dossier = {"summary": "critical outage", "risks": [{"severity": "critical"}]}
        decision = await probe.arun(
            task="restore service",
            dossier=dossier,
            execution_metadata=metadata,
        )
        assert decision.mode == "escalate"
        assert decision.confidence == pytest.approx(0.34, rel=1e-2)
        assert "learning_history_sparse" in decision.raw["evidence"]
        assert "recent_scores_below_threshold" in decision.raw["evidence"]

    asyncio.run(runner())


def test_areview_plan_refreshes_cache_on_escalation():
    async def runner() -> None:
        ExecutionContext.get().reset()
        ExecutionContext.get().metadata["adaptive"] = {"active_mode": "coach"}
        config = TeacherConfig(
            llm=_gpt5_params(),
            max_review_tokens=1024,
            plan_cache_seconds=60,
            guidance_max_tokens=512,
            validation_max_tokens=512,
        )
        prompts = RewrittenTeacherPrompts(
            plan_review="Respond with JSON containing a 'steps' array.",
            validation="Return JSON {\"valid\": bool, \"guidance\": str | null}.",
            guidance="Return short guidance.",
        )
        teacher = Teacher(config, prompts)
        plan = Plan(steps=[Step(id=1, description="draft summary", depends_on=[])])
        teacher._client = _StaticResponseClient(
            {"steps": [{"id": 1, "description": "coach plan", "depends_on": [], "tool": None, "tool_params": None}]}
        )
        reviewed_coach = await teacher.areview_plan("Summarize Atlas", plan)
        assert reviewed_coach.steps[0].description == "coach plan"
        ExecutionContext.get().metadata["adaptive"]["active_mode"] = "escalate"
        teacher._client = _StaticResponseClient(
            {"steps": [{"id": 1, "description": "escalated plan", "depends_on": [], "tool": None, "tool_params": None}]}
        )
        reviewed_escalate = await teacher.areview_plan("Summarize Atlas", plan)
        assert reviewed_escalate.steps[0].description == "escalated plan"

    asyncio.run(runner())


    asyncio.run(runner())


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
