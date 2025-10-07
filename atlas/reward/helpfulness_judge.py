"""Helpfulness judge implementation."""

from __future__ import annotations

import json
from typing import Dict, Sequence

from atlas.reward.judge import Judge, JudgeContext
from atlas.reward.judge_prompts import HELPFULNESS_PROMPT
from atlas.utils.llm_client import LLMClient


class HelpfulnessJudge(Judge):
    def __init__(self, client: LLMClient) -> None:
        super().__init__("helpfulness", client)

    async def ajudge(self, context: JudgeContext):  # pragma: no cover - evaluator orchestrates outcomes
        raise NotImplementedError("Evaluator controls helpfulness judge aggregation")

    def _build_messages(self, context: JudgeContext) -> Sequence[Dict[str, str]]:
        system_prompt = HELPFULNESS_PROMPT.format(
            task=context.task,
            step_id=context.step.id,
            step_description=context.step.description,
            guidance=json.dumps(list(context.guidance or []), ensure_ascii=False, indent=2),
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
            "guidance_history": list(context.guidance or []),
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
            sample_text = "Tier-1 judges produced no parsable output; escalating directly to you."

        reason_text = (
            f"Escalation reason: {escalation_reason}\n" if escalation_reason else "Escalation triggered by disagreement or high uncertainty.\n"
        )

        return (
            "SYSTEM: You are the Atlas Teaching Helpfulness Arbiter. Tier-1 judges are uncertain about the guidance"
            " impact. Review their work and deliver the final judgement.\n\n"
            f"{reason_text}Task: {context.task}\n"
            f"Step Description: {context.step.description}\n"
            f"Student Execution Trace: {context.trace}\n"
            f"Student Output: {context.output}\n\n"
            "Tier-1 evaluations (principles, score, uncertainty, rationale):\n"
            f"{sample_text}\n\n"
            "Output JSON with keys principles, score, rationale, uncertainty. Explain how you reconciled conflicts,"
            " identified missed opportunities in the guidance, and whether escalation to a human teacher is required."
        )
