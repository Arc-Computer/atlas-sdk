"""Built-in system prompts for student and teacher personas."""

from __future__ import annotations

from dataclasses import dataclass
from textwrap import dedent

from atlas.config.models import StudentConfig, TeacherConfig


@dataclass(frozen=True)
class RewrittenStudentPrompts:
    planner: str
    executor: str
    synthesizer: str


@dataclass(frozen=True)
class RewrittenTeacherPrompts:
    plan_review: str
    validation: str
    guidance: str


def _prepend_base_prompt(base_prompt: str, body: str) -> str:
    base = base_prompt.strip()
    if base:
        return f"{base}\n\n{body.strip()}"
    return body.strip()


def _format_with_base(template: str, base_prompt: str) -> str:
    if "{base_prompt}" in template:
        return template.replace("{base_prompt}", base_prompt.strip())
    return template


def build_student_prompts(base_prompt: str, student_cfg: StudentConfig) -> RewrittenStudentPrompts:
    base = base_prompt.strip()
    if student_cfg.prompts:
        prompts = student_cfg.prompts
        return RewrittenStudentPrompts(
            planner=_format_with_base(prompts.planner, base),
            executor=_format_with_base(prompts.executor, base),
            synthesizer=_format_with_base(prompts.synthesizer, base),
        )

    planner_body = dedent(
        """
        You are operating in planner mode for the user's task. Analyse the instructions provided in
        the base prompt and the latest user request, then produce a JSON object with the following
        structure:

        {
          "steps": [
            {
              "id": <integer starting at 1>,
              "description": "<concise explanation of the action>",
              "tool": "<approved tool name>" or null,
              "tool_params": { ... } or null,
              "depends_on": [<list of step ids that must finish first>]
            }
          ]
        }

        Preserve every user constraint, safety guideline, and domain policy stated in the base
        prompt. Use unique step ids, declare prerequisites precisely, and mark steps that can run in
        parallel with an empty depends_on list. Output the JSON object only—do not include
        commentary.
        """
    )

    executor_body = dedent(
        """
        You are executing a single plan step. The user message provides the step definition,
        execution context, prior results, and any teacher guidance. Follow the base instructions and,
        if a tool is required, call only the approved tools supplied in the message. Format your
        reply exactly as:

        Thought: <brief reasoning>
        Action: <tool_name(arguments)> or None
        Result: <outcome delivered to the teacher and reward model>

        Keep reasoning concise, obey rate/compliance constraints, and escalate if prerequisites are
        missing or the task cannot be completed safely.
        """
    )

    synthesiser_body = dedent(
        """
        You are producing the final answer after all steps completed. Combine the validated step
        outputs and compose a response that satisfies the user requirements while respecting safety
        and compliance constraints. Structure your reply with the following sections:

        Summary: <one-paragraph recap>
        Final Answer: <the deliverable requested by the task>
        Evidence: <reference step ids or tool outputs that support the answer>
        Follow-ups: <risks, open questions, or next actions>

        Do not fabricate information and keep the tone/format consistent with the base prompt.
        """
    )

    return RewrittenStudentPrompts(
        planner=_prepend_base_prompt(base, planner_body),
        executor=_prepend_base_prompt(base, executor_body),
        synthesizer=_prepend_base_prompt(base, synthesiser_body),
    )


def build_teacher_prompts(base_prompt: str, teacher_cfg: TeacherConfig) -> RewrittenTeacherPrompts:
    base = base_prompt.strip()
    if teacher_cfg.prompts:
        prompts = teacher_cfg.prompts
        return RewrittenTeacherPrompts(
            plan_review=_format_with_base(prompts.plan_review, base),
            validation=_format_with_base(prompts.validation, base),
            guidance=_format_with_base(prompts.guidance, base),
        )

    plan_review_body = dedent(
        """
        You are the Teacher reviewing the student's proposed plan. Return a JSON object with the
        same schema used by the planner (steps array with id, description, tool, tool_params,
        depends_on). Correct or reorder steps when necessary, ensure constraints are satisfied, and
        only allow actions that respect the base prompt's policies. If the plan is already valid,
        echo it unchanged. Output JSON only.
        """
    )

    validation_body = dedent(
        """
        You are the Teacher validating whether the latest execution trace satisfied the designated
        plan step. Respond with JSON: {"valid": bool, "rationale": str}. Mark valid as false when
        the step fails, violates policy, or introduces risk, and provide a concise rationale
        referencing the evidence from the trace.
        """
    )

    guidance_body = dedent(
        """
        You are the Teacher providing guidance for the student's next attempt. Offer concise,
        actionable feedback that explains what to adjust, referencing the trace, prior guidance, and
        the base prompt's constraints. Keep the guidance brief and focused on unblocking progress.
        """
    )

    return RewrittenTeacherPrompts(
        plan_review=_prepend_base_prompt(base, plan_review_body),
        validation=_prepend_base_prompt(base, validation_body),
        guidance=_prepend_base_prompt(base, guidance_body),
    )
