"""Process quality judge."""

from __future__ import annotations

import json
from typing import Dict
from typing import Sequence

from atlas.config.models import JudgeConfig
from atlas.reward.judge import Judge
from atlas.reward.judge import JudgeContext
from atlas.reward.judge_prompts import PROCESS_PROMPT


class ProcessJudge(Judge):
    def __init__(self, config: JudgeConfig) -> None:
        super().__init__(config)

    def _build_messages(self, context: JudgeContext) -> Sequence[Dict[str, Any]]:
        payload = {
            "task": context.task,
            "step": context.step.model_dump(),
            "trace": context.trace,
            "output": context.output,
            "attempt": context.attempt,
        }
        return [
            {"role": "system", "content": PROCESS_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
