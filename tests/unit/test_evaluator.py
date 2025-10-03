import asyncio

import pytest

pytest.importorskip("google")

from atlas.config.models import JudgeConfig, JudgeKind, LLMParameters, LLMProvider, RIMConfig
from atlas.reward.evaluator import Evaluator
from atlas.reward.judge import JudgeContext
from atlas.types import Step


def _gemini_params(model: str, max_tokens: int) -> LLMParameters:
    return LLMParameters(
        provider=LLMProvider.GOOGLE,
        model=f"gemini/{model}",
        api_key_env="GEMINI_API_KEY",
        temperature=0.2,
        max_output_tokens=max_tokens,
    )


def _attach_capture(client) -> dict[str, str]:
    captured: dict[str, str] = {}
    original = client.acomplete

    async def traced(messages, response_format=None, overrides=None):
        response = await original(messages, response_format, overrides)
        captured["content"] = response.content
        return response

    client.acomplete = traced
    return captured


def test_evaluator_live_gemini_scores():
    async def runner() -> None:
        config = RIMConfig(
            judges=[
                JudgeConfig(
                    identifier="process",
                    kind=JudgeKind.PROCESS,
                    weight=0.5,
                    principles=["Follow the plan"],
                    llm=_gemini_params("gemini-2.5-flash", 512),
                    max_tokens=512,
                ),
                JudgeConfig(
                    identifier="helpfulness",
                    kind=JudgeKind.HELPFULNESS,
                    weight=0.5,
                    principles=["Stay on topic"],
                    llm=_gemini_params("gemini-2.5-flash", 512),
                    max_tokens=512,
                ),
            ],
            temperatures=[0.0],
            variance_threshold=1.0,
            uncertainty_threshold=1.0,
            arbiter=_gemini_params("gemini-2.5-pro", 512),
            success_threshold=0.5,
            retry_threshold=0.3,
            aggregation_strategy="weighted_mean",
        )
        evaluator = Evaluator(config)
        judge_captures = [_attach_capture(judge._client) for judge in evaluator._judges]
        arbiter_capture = _attach_capture(evaluator._arbiter_client)
        context = JudgeContext(
            task="Produce a brief assessment of Atlas progress",
        step=Step(id=1, description="Summarize the latest milestone", depends_on=[]),
            trace="Student described achievements and cited benchmarks.",
            output="Atlas delivered new GDPval metrics with detailed justification.",
            guidance=[],
        )
        try:
            result = await evaluator.ajudge(context)
        except Exception:
            for idx, capture in enumerate(judge_captures, start=1):
                raw = capture.get("content", "")
                if raw:
                    print(f"Judge {idx} raw response: {raw}")
            raw = arbiter_capture.get("content", "")
            if raw:
                print(f"Arbiter raw response: {raw}")
            raise
        score = result.get("score")
        if not isinstance(score, (int, float)):
            for capture in judge_captures:
                raw = capture.get("content", "")
                if raw:
                    print(f"Judge raw response: {raw}")
            raw = arbiter_capture.get("content", "")
            if raw:
                print(f"Arbiter raw response: {raw}")
            pytest.fail("RIM evaluator did not return a numeric score")
        for entry, capture in zip(result.get("judges", []), judge_captures, strict=False):
            principles = entry.get("principles")
            rationale = entry.get("rationale")
            samples = entry.get("samples")
            if not principles or not rationale or not samples:
                raw = capture.get("content", "")
                if raw:
                    print(f"Judge response payload: {raw}")
                pytest.fail("Judge response missing principles, rationale, or samples")

    asyncio.run(runner())
