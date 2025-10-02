"""Base judge implementations for the RIM evaluator."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Sequence

from atlas.config.models import JudgeConfig
from atlas.types import Step
from atlas.utils.llm_client import LLMClient


@dataclass
class JudgeContext:
    task: str
    step: Step
    trace: str
    output: str
    attempt: int = 1
    prior_results: Dict[int, str] | None = None


@dataclass
class JudgeResult:
    identifier: str
    score: float
    rationale: str
    principles: Sequence[str]
    raw: Any


class Judge:
    def __init__(self, config: JudgeConfig) -> None:
        self.identifier = config.identifier
        self.weight = config.weight
        self.principles = list(config.principles)
        self._default_principles = list(config.principles)
        self._client = LLMClient(config.llm)

    async def ajudge(self, context: JudgeContext) -> JudgeResult:
        messages = self._build_messages(context)
        response = await self._client.acomplete(messages, response_format={"type": "json_object"})
        payload = json.loads(response.content)
        score = float(payload.get("score", 0.0))
        rationale = payload.get("rationale", "")
        runtime_principles = self._extract_principles(payload)
        if not runtime_principles:
            runtime_principles = self._default_principles
        return JudgeResult(
            identifier=self.identifier,
            score=max(0.0, min(score, 1.0)),
            rationale=rationale,
            principles=runtime_principles,
            raw=response.raw,
        )

    def judge(self, context: JudgeContext) -> JudgeResult:
        return asyncio.run(self.ajudge(context))

    def _extract_principles(self, payload: Dict[str, Any]) -> List[str]:
        values = payload.get("principles")
        principles: List[str] = []
        if isinstance(values, str):
            principles.append(values)
        elif isinstance(values, list):
            for item in values:
                if isinstance(item, str):
                    principles.append(item)
                elif isinstance(item, dict):
                    if item.get("name"):
                        principles.append(str(item["name"]))
                    else:
                        principles.append(json.dumps(item))
        return principles

    def _build_messages(self, context: JudgeContext) -> Sequence[Dict[str, Any]]:
        raise NotImplementedError
