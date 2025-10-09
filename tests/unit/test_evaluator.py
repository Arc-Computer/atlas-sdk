import asyncio
import json
from typing import Dict

from atlas.config.models import LLMParameters, RIMConfig
from atlas.evaluation.evaluator import Evaluator
from atlas.evaluation.judges.base import Judge, JudgeContext
from atlas.types import Step
from atlas.utils.llm_client import LLMResponse
from atlas.runtime.orchestration.execution_context import ExecutionContext


class _StubClient:
    def __init__(self, scores_by_temperature: Dict[float, float]) -> None:
        self._scores = scores_by_temperature

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
        raw_judges = (result.raw or {}).get("judges", [])
        assert raw_judges, "Raw payload should include judge details"
        for raw_entry in raw_judges:
            assert raw_entry.get("reasoning") is not None
        assert 0 <= result.score <= 1

    asyncio.run(runner())
