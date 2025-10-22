import asyncio
import json
from typing import Dict

from atlas.config.models import LLMParameters, RIMConfig
from atlas.evaluation.evaluator import Evaluator
from atlas.evaluation.judges.base import Judge, JudgeContext, JudgeSample
from atlas.types import Step
from atlas.utils.llm_client import LLMResponse
from atlas.runtime.orchestration.execution_context import ExecutionContext


class _StubClient:
    def __init__(self, scores_by_temperature: Dict[float, float]) -> None:
        self._scores = scores_by_temperature
        self.model = "stub-model"

    async def acomplete(self, messages, response_format=None, overrides=None):
        temperature = (overrides or {}).get("temperature", 0.0)
        score = self._scores.get(temperature, 0.75)
        payload = {
            "principles": [
                {"name": "P1", "weight": 0.5, "description": "execution quality"},
                {"name": "P2", "weight": 0.5, "description": "safety compliance"},
            ],
            "score": score,
            "rationale": f"scored at {score:.2f}",
            "uncertainty": 0.1,
        }
        reasoning = {"thinking_blocks": [{"type": "analysis", "text": f"confidence {score:.2f}"}]}
        return LLMResponse(content=json.dumps(payload), raw={}, reasoning=reasoning)


class _QueueingClient:
    def __init__(self) -> None:
        self.model = "stub-model"

    async def acomplete(self, messages, response_format=None, overrides=None):
        context = ExecutionContext.get()
        queue = context.metadata.setdefault("_llm_reasoning_queue", [])
        queue.append({"origin": ("reward", "sample"), "payload": {"thought": "analysis"}})
        payload = {
            "principles": [],
            "score": 0.85,
            "rationale": "consistent",
            "uncertainty": 0.1,
        }
        reasoning = {"thinking_blocks": [{"type": "analysis", "text": "queue merged"}]}
        return LLMResponse(content=json.dumps(payload), raw={}, reasoning=reasoning)


class _StubJudge(Judge):
    def _build_messages(self, context: JudgeContext):
        return [{"role": "system", "content": "judge"}]

    async def asample(self, context: JudgeContext, temperature: float):
        response = await self._client.acomplete(
            self._build_messages(context),
            response_format={"type": "json_object"},
            overrides={"temperature": temperature},
        )
        payload = json.loads(response.content)
        exec_context = ExecutionContext.get()
        queue = exec_context.metadata.get("_llm_reasoning_queue", [])
        matched = []
        remaining = []
        for entry in queue or []:
            if entry.get("origin") == ("reward", "sample"):
                matched.append(entry.get("payload") or {})
            else:
                remaining.append(entry)
        exec_context.metadata["_llm_reasoning_queue"] = remaining
        reasoning = response.reasoning or {}
        if matched:
            reasoning = {"response": reasoning, "queue": matched} if reasoning else {"queue": matched}
        return JudgeSample(
            score=payload.get("score", 0.0),
            rationale=payload.get("rationale", ""),
            principles=payload.get("principles") or [],
            uncertainty=payload.get("uncertainty", 0.0),
            temperature=temperature,
            reasoning=reasoning or None,
        )


class _ExplodingJudge(Judge):
    def __init__(self, identifier: str) -> None:
        super().__init__(identifier, _StubClient({}))

    async def asample(self, context: JudgeContext, temperature: float):
        raise AssertionError("Override should short-circuit before sampling judges")

    async def ajudge(self, context: JudgeContext):
        raise AssertionError("Override should short-circuit before direct judge invocation")

    def build_meta_prompt(self, context: JudgeContext, samples, escalation_reason):
        raise AssertionError("Override should short-circuit before escalation")

    def _build_messages(self, context: JudgeContext):
        return []


def test_judge_asample_merges_queue_reasoning():
    async def runner() -> None:
        ExecutionContext.get().reset()
        client = _QueueingClient()
        judge = _StubJudge("process", client)
        context = JudgeContext(
            task="demo",
            step=Step(id=1, description="collect", depends_on=[]),
            trace="log",
            output="result",
        )
        sample = await judge.asample(context, 0.2)
        assert sample is not None
        assert sample.reasoning is not None
        assert sample.reasoning["queue"][0]["thought"] == "analysis"
        assert ExecutionContext.get().metadata.get("_llm_reasoning_queue") == []

    asyncio.run(runner())


def test_evaluator_combines_judge_scores():
    async def runner() -> None:
        config = RIMConfig(
            small_model=LLMParameters(model="stub"),
            large_model=LLMParameters(model="arbiter"),
            active_judges={"process": True, "helpfulness": True},
            variance_threshold=1.0,
            uncertainty_threshold=1.0,
            parallel_workers=2,
        )
        small_client = _StubClient({0.2: 0.8, 0.5: 0.7, 0.8: 0.6})
        large_client = _StubClient({0.3: 0.75})
        evaluator = Evaluator(config, small_client=small_client, large_client=large_client)
        context = JudgeContext(
            task="Summarise Atlas milestone",
            step=Step(id=1, description="Compile results", depends_on=[]),
            trace="Student gathered data and produced a summary.",
            output="Consolidated metrics with references.",
            guidance=["Focus on GDPval metrics"],
        )
        ExecutionContext.get().reset()
        result = await evaluator.ajudge(context)
        assert isinstance(result.score, float)
        assert result.judges, "Expected per-judge breakdown"
        for entry in result.judges:
            assert entry.principles, "Principles should be present"
            assert isinstance(entry.score, float)
        raw_payload = result.raw or {}
        raw_samples = raw_payload.get("samples", [])
        assert raw_samples, "Raw payload should include sample details"
        for raw_entry in raw_samples:
            assert "score" in raw_entry and "uncertainty" in raw_entry
        assert 0 <= result.score <= 1

    asyncio.run(runner())


def test_evaluator_accepts_precomputed_reward():
    async def runner() -> None:
        config = RIMConfig(
            small_model=LLMParameters(model="stub"),
            large_model=LLMParameters(model="arbiter"),
            active_judges={"process": True},
            variance_threshold=1.0,
            uncertainty_threshold=1.0,
            parallel_workers=1,
        )
        evaluator = Evaluator(config, small_client=_StubClient({}), large_client=_StubClient({}))
        context = JudgeContext(
            task="certification",
            step=Step(id=1, description="Check answer", depends_on=[]),
            trace="",
            output="",
            reward_override={"score": 0.92, "rationale": "teacher certified", "judges": []},
        )
        ExecutionContext.get().reset()
        result = await evaluator.ajudge(context)
        assert result.score == 0.92
        assert result.raw and result.raw.get("score") == 0.92

    asyncio.run(runner())


def test_evaluator_reward_override_short_circuits_judges():
    async def runner() -> None:
        config = RIMConfig(
            small_model=LLMParameters(model="stub"),
            large_model=LLMParameters(model="arbiter"),
            active_judges={"process": True},
            variance_threshold=1.0,
            uncertainty_threshold=1.0,
            parallel_workers=1,
        )
        evaluator = Evaluator(config, small_client=_StubClient({}), large_client=_StubClient({}))
        evaluator._judges = [_ExplodingJudge("explode")]  # type: ignore[attr-defined]
        context = JudgeContext(
            task="reuse certification verdict",
            step=Step(id=99, description="final validate", depends_on=[]),
            trace="",
            output="",
            reward_override={"score": 0.88, "rationale": "teacher certified", "judges": []},
        )
        ExecutionContext.get().reset()
        result = await evaluator.ajudge(context)
        assert result.score == 0.88
        assert result.raw and result.raw.get("score") == 0.88

    asyncio.run(runner())
