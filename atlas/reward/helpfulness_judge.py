"""Helpfulness judge."""

from __future__ import annotations

import json
from typing import Dict
from typing import Sequence

from atlas.config.models import JudgeConfig
from atlas.reward.judge import Judge
from atlas.reward.judge import JudgeContext
from atlas.reward.judge_prompts import HELPFULNESS_PROMPT


class HelpfulnessJudge(Judge):
    def __init__(self, config: JudgeConfig) -> None:
        super().__init__(config)

    def _build_messages(self, context: JudgeContext) -> Sequence[Dict[str, str]]:
        payload = {
            "task": context.task,
            "step": context.step.model_dump(),
            "output": context.output,
            "trace_summary": context.trace,
        }
        return [
            {"role": "system", "content": HELPFULNESS_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
