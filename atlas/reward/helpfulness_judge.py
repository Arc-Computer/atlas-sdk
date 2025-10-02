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
                f"Evaluation {i+1}:\n"
                f"Principles: {json.dumps(sample.principles)}\n"
                f"Score: {sample.score:.2f}\n"
                f"Uncertainty: {sample.uncertainty:.2f}\n"
                f"Rationale: {sample.rationale}"
                for i, sample in enumerate(samples)
            )
        else:
            sample_text = "Tier-1 judges did not produce usable outputs."

        reason_text = f"Escalation reason: {escalation_reason}\n" if escalation_reason else "Escalation triggered by high variance / uncertainty.\n"

        guidance_text = json.dumps(context.guidance or [], ensure_ascii=False)

        return (
            "You are the escalation arbiter for teaching helpfulness. Tier-1 judges disagreed or were uncertain"
            " about how effective the teacher's guidance was.\n\n"
            f"{reason_text}Task: {context.task}\n"
            f"Teacher Guidance: {guidance_text}\n"
            f"Student Execution Trace: {context.trace}\n"
            f"Student Output: {context.output}\n\n"
            "Tier-1 evaluations (principles, score, uncertainty, rationale):\n"
            f"{sample_text}\n\n"
            "Instructions:\n"
            "1. Analyse the principles used by the tier-1 judges. Identify agreements, conflicts, or omissions.\n"
            "2. Reuse the strongest principles or create new ones that better capture helpfulness in this context.\n"
            "3. Re-evaluate the guidance versus the student's outcome using your chosen principles.\n"
            "4. Provide a final score in [0,1] with rationale tied to each selected principle.\n"
            "5. Report an uncertainty value in [0,1].\n\n"
            "Return JSON {{\"principles\": [{{\"name\": str, \"weight\": float, \"description\": str}}],"
            " \"score\": float, \"rationale\": str, \"uncertainty\": float}}."
        )
