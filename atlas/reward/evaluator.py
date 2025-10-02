"""RIM evaluator orchestrating multiple judges."""

from __future__ import annotations

import asyncio
from statistics import fmean
from typing import Any
from typing import Dict
from typing import List

from atlas.config.models import JudgeConfig
from atlas.config.models import JudgeKind
from atlas.config.models import RIMConfig
from atlas.reward.helpfulness_judge import HelpfulnessJudge
from atlas.reward.judge import Judge
from atlas.reward.judge import JudgeContext
from atlas.reward.judge import JudgeResult
from atlas.reward.process_judge import ProcessJudge


_JUDGE_FACTORY: Dict[JudgeKind, type[Judge]] = {
    JudgeKind.PROCESS: ProcessJudge,
    JudgeKind.HELPFULNESS: HelpfulnessJudge,
    JudgeKind.CUSTOM: ProcessJudge,
}


class Evaluator:
    def __init__(self, config: RIMConfig) -> None:
        self._config = config
        self._judges = [self._build_judge(judge_config) for judge_config in config.judges]

    async def ajudge(self, context: JudgeContext) -> Dict[str, Any]:
        results = await asyncio.gather(*(judge.ajudge(context) for judge in self._judges))
        aggregated = self._aggregate(results)
        return {
            "score": aggregated,
            "judges": [self._serialize_result(result, judge) for result, judge in zip(results, self._judges)],
        }

    def judge(self, context: JudgeContext) -> Dict[str, Any]:
        return asyncio.run(self.ajudge(context))

    def _aggregate(self, results: List[JudgeResult]) -> float:
        if not results:
            return 0.0
        if self._config.aggregation_strategy == "minimum":
            return min(result.score for result in results)
        weights = [result.score * judge.weight for result, judge in zip(results, self._judges)]
        total_weight = sum(judge.weight for judge in self._judges)
        if total_weight == 0:
            return fmean(result.score for result in results)
        return sum(weights) / total_weight

    def _build_judge(self, config: JudgeConfig) -> Judge:
        factory = _JUDGE_FACTORY.get(config.kind)
        if factory is None:
            raise ValueError(f"Unsupported judge kind {config.kind}")
        return factory(config)

    def _serialize_result(self, result: JudgeResult, judge: Judge) -> Dict[str, Any]:
        principles = list(result.principles) if result.principles else list(judge.principles)
        return {
            "identifier": result.identifier,
            "score": result.score,
            "rationale": result.rationale,
            "principles": principles,
        }
