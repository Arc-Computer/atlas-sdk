import asyncio
import json

import pytest

from atlas.config.models import JudgeConfig, JudgeKind, LLMParameters, RIMConfig
from atlas.reward.evaluator import Evaluator
from atlas.reward.judge import JudgeContext
from atlas.utils.llm_client import LLMResponse
from atlas.types import Step


class QueueLLMClient:
    def __init__(self, *_args, **_kwargs):
        self.responses = []

    async def acomplete(self, messages, response_format=None):
        payload = self.responses.pop(0)
        return LLMResponse(content=json.dumps(payload), raw={})


def test_evaluator_weighted_mean(monkeypatch):
    client = QueueLLMClient()
    monkeypatch.setattr("atlas.reward.judge.LLMClient", lambda *_: client)
    config = RIMConfig(
        judges=[
            JudgeConfig(identifier="process", kind=JudgeKind.PROCESS, weight=0.7, principles=[], llm=LLMParameters(model="stub")),
            JudgeConfig(identifier="helpful", kind=JudgeKind.HELPFULNESS, weight=0.3, principles=[], llm=LLMParameters(model="stub")),
        ],
        success_threshold=0.7,
        retry_threshold=0.6,
        aggregation_strategy="weighted_mean",
    )
    evaluator = Evaluator(config)
    client.responses.append({"score": 0.8, "rationale": "score=0.8", "principles": ["process"]})
    client.responses.append({"score": 0.2, "rationale": "score=0.2", "principles": [{"name": "helpful"}]})
    context = JudgeContext(
        task="task",
        step=Step(id=1, description="desc", depends_on=[], estimated_time="1m"),
        trace="trace",
        output="output",
    )
    result = asyncio.run(evaluator.ajudge(context))
    assert pytest.approx(result["score"], rel=1e-5) == 0.62
    assert len(result["judges"]) == 2
    assert result["judges"][0]["principles"] == ["process"]
    assert result["judges"][1]["principles"] == ["helpful"]
