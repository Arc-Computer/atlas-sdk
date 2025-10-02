"""Helpfulness judge implementation."""

from __future__ import annotations

import json
from typing import Dict, Sequence

from atlas.config.models import JudgeConfig
from atlas.reward.judge import Judge, JudgeContext
from atlas.reward.judge_prompts import HELPFULNESS_PROMPT


class HelpfulnessJudge(Judge):
    def __init__(self, config: JudgeConfig) -> None:
        super().__init__(config)

    async def ajudge(self, context: JudgeContext):  # pragma: no cover - evaluator drives outcomes
        raise NotImplementedError("HelpfulnessJudge outcomes are computed by the evaluator")

    def _build_messages(self, context: JudgeContext) -> Sequence[Dict[str, str]]:
        system_prompt = HELPFULNESS_PROMPT.format(
            task=context.task,
            teacher_trace=json.dumps(context.guidance or [], ensure_ascii=False, indent=2),
            student_trace=context.trace,
            student_output=context.output,
            prior_results=json.dumps(context.prior_results or {}, ensure_ascii=False, indent=2),
        )
        payload = {
            "task": context.task,
            "step": context.step.model_dump(),
            "execution_trace": context.trace,
            "final_output": context.output,
            "prior_step_results": context.prior_results or {},
        }
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]

    def build_meta_prompt(self, context: JudgeContext, samples, escalation_reason: str | None) -> str:
        if samples:
            sample_text = "\n\n".join(
                f"Sample {i+1}:\nPrinciples: {json.dumps(sample.principles)}\n"
                f"Score: {sample.score:.2f}\nUncertainty: {sample.uncertainty:.2f}\n"
                f"Rationale: {sample.rationale}"
                for i, sample in enumerate(samples)
            )
        else:
            sample_text = "No valid tier-1 samples were produced."

        reason_text = f"Reason for escalation: {escalation_reason}\n" if escalation_reason else ""

        return (
            "You are the escalation arbiter for teaching helpfulness. Review the teacher guidance and student outcome"
            " to issue a final judgment.\n\n"
            f"Task: {context.task}\n"
            f"Teacher Guidance: {json.dumps(context.guidance or [], ensure_ascii=False)}\n"
            f"Student Execution Trace: {context.trace}\n"
            f"Student Output: {context.output}\n\n"
            f"{reason_text}Tier-1 Samples:\n{sample_text}\n\n"
            "Return JSON {{\"principles\": [{{\"name\": str, \"weight\": float, \"description\": str}}],"
            " \"score\": float, \"rationale\": str, \"uncertainty\": float}}."
        )
