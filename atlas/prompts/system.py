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
        Think through how you would approach this task. What steps would you take to complete it?

        Consider:
        - What needs to be done, in what order
        - Which steps depend on others completing first
        - What tools or capabilities you would use for each step

        Express your plan as a structured breakdown of the work.
        """
    )
    planner_schema = dedent(
        """
        JSON schema reference for planner responses:
        {
          "steps": [
            {
              "id": integer,
              "description": string,
              "tool": string | null,
              "tool_params": object | null,
              "depends_on": [integer]
            }
          ]
        }
        """
    )

    executor_body = dedent(
        """
        Complete the step described below using the provided context and tools.

        Report what you did and what results you produced.
        """
    )
    executor_schema = dedent(
        """
        JSON schema reference for executor responses:
        {
          "status": string,
          "result": {
            "deliverable": any,
            "artifacts": object | null
          },
          "deliverable": any | null,
          "artifacts": object | null,
          "reason": string | null,
          "text": string | null
        }
        """
    )

    synthesiser_body = dedent(
        """
        The work has been completed. Based on the steps taken and their results, provide the final answer to the original request.
        """
    )
    synthesiser_schema = dedent(
        """
        JSON schema reference for synthesis responses:
        {
          "final_answer": string,
          "supporting_evidence": [string] | null
        }
        """
    )

    planner_prompt = "\n\n".join([planner_body.strip(), planner_schema.strip()])
    executor_prompt = "\n\n".join([executor_body.strip(), executor_schema.strip()])
    synthesiser_prompt = "\n\n".join([synthesiser_body.strip(), synthesiser_schema.strip()])

    return RewrittenStudentPrompts(
        planner=_prepend_base_prompt(base, planner_prompt),
        executor=_prepend_base_prompt(base, executor_prompt),
        synthesizer=_prepend_base_prompt(base, synthesiser_prompt),
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
        Review the student's proposed plan for this task.

        You have the same knowledge, tools, and context as the student. Your focus is assessing whether their plan adequately addresses the user's request.

        Evaluate:
        1. Completeness: Does the plan cover all parts of the task?
        2. Feasibility: Can the plan be executed with available tools and capabilities?
        3. Structure: Are dependencies between steps clear and logical?
        4. Scope: Is the plan appropriately sized - neither missing steps nor overly complex?

        Provide your assessment along with the corrected plan if changes are needed (otherwise return the original plan) and call out any concerns about gaps or risks.
        """
    )
    plan_review_schema = dedent(
        """
        JSON schema reference for plan review responses:
        {
          "steps": [
            {
              "id": integer,
              "description": string,
              "tool": string | null,
              "tool_params": object | null,
              "depends_on": [integer]
            }
          ],
          "concerns": [string] | null
        }
        """
    )

    validation_body = dedent(
        """
        Validate whether the student's execution is acceptable.

        You have the same knowledge, tools, and context as the student. Your focus is assessing whether their work correctly completes the assigned step or task.

        Evaluate the student's output:
        - Does it correctly complete what the step or task requested?
        - Is it accurate given the context and requirements?
        - Are there critical errors or failures?

        Make your validation decision:
        - valid=true: The output is acceptable, proceed to next step or task
        - valid=false: The output needs correction

        If validation fails (valid=false), provide guidance for correction:
        - Be specific about what's wrong
        - Reference the exact issue in the student's output
        - State what needs to be fixed
        - Keep guidance focused and actionable

        Provide your validation decision and, if the output is invalid, specific correction guidance.
        """
    )
    validation_schema = dedent(
        """
        JSON schema reference for validation responses:
        {
          "valid": bool,
          "guidance": string | null
        }
        """
    )

    guidance_body = validation_body
    guidance_schema = validation_schema

    return RewrittenTeacherPrompts(
        plan_review=_prepend_base_prompt(base, "\n\n".join([plan_review_body.strip(), plan_review_schema.strip()])),
        validation=_prepend_base_prompt(base, "\n\n".join([validation_body.strip(), validation_schema.strip()])),
        guidance=_prepend_base_prompt(base, "\n\n".join([guidance_body.strip(), guidance_schema.strip()])),
    )
