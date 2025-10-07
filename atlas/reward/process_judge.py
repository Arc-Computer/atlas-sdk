"""Process-quality judge implementation."""

from __future__ import annotations

import json
from typing import Dict, Sequence

from atlas.reward.judge import Judge, JudgeContext
from atlas.reward.judge_prompts import PROCESS_PROMPT
from atlas.utils.llm_client import LLMClient


class ProcessJudge(Judge):
    def __init__(self, client: LLMClient) -> None:
        super().__init__("process", client)

    async def ajudge(self, context: JudgeContext):
        raise NotImplementedError("Evaluator controls process judge aggregation")

    def _build_messages(self, context: JudgeContext) -> Sequence[Dict[str, str]]:
        system_prompt = PROCESS_PROMPT.format(
            task=context.task,
            step_id=context.step.id,
            step_description=context.step.description,
            dependencies=json.dumps(context.step.depends_on, ensure_ascii=False),
            attempt=context.attempt,
            guidance=json.dumps(list(context.guidance or []), ensure_ascii=False, indent=2),
            prior_results=json.dumps(context.prior_results or {}, ensure_ascii=False, indent=2),
            student_trace=context.trace,
            student_output=context.output,
        )
        payload = {
            "step": context.step.model_dump(),
            "execution_trace": context.trace,
            "final_output": context.output,
            "attempt": context.attempt,
            "prior_results": context.prior_results or {},
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
            "SYSTEM: You are the Atlas Process Arbiter. Tier-1 judges disagreed about this execution attempt."
            " Review their evaluations and deliver the final judgement.\n\n"
            f"{reason_text}Task: {context.task}\n"
            f"Step ID: {context.step.id}\n"
            f"Step Description: {context.step.description}\n"
            f"Execution Trace: {context.trace}\n"
            f"Student Output: {context.output}\n\n"
            "Tier-1 evaluations (principles, score, uncertainty, rationale):\n"
            f"{sample_text}\n\n"
            "Output JSON with keys principles, score, rationale, uncertainty. Apply your own high-fidelity principles"
            " when synthesising the final result. Ensure the rationale explains how you reconciled conflicts and"
            " addressed compliance or safety concerns."
        )
