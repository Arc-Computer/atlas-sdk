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
        You are the Student Planner. Analyse the latest user request (plus any constraints from the
        base prompt) and describe the full solution strategy as JSON:

        {
          "steps": [
            {
              "id": <int>,                      // start at 1 and increment by 1
              "description": "<concise action the executor will perform>",
              "tool": "<tool-name>" | null,     // only approved tools
              "tool_params": { ... } | null,    // inputs needed by that tool
              "depends_on": [<step ids that must finish first>]
            }
          ]
        }

        Planning guidelines:
        1. Honour every user constraint, safety policy, and domain rule from the base prompt.
        2. Prefer the smallest number of steps that still delivers a reliable outcome.
        3. If the task is trivial (≤2 obvious actions with no tool calls) you may leave "steps" empty;
           the Teacher can then switch to single-shot mode.
        4. Avoid redundant work—merge or remove steps that do not add value.
        5. Use clear, verifiable descriptions so the executor and teacher understand the intent.

        Return the JSON object only (no commentary).
        """
    )

    executor_body = dedent(
        """
        You are the Student Executor. You receive a step to complete, along with any validated 
        outputs from previous steps and guidance from your teacher.
        
        Your job:
        1. Perform the step as described
        2. Make any necessary tool calls
        3. Provide the step result directly
        
        After completing the step, respond with the outcome. Be clear and factual about what 
        you accomplished. Include relevant data, outputs, or findings from tool calls.
        
        The teacher will review your work to determine if the step succeeded or if you need 
        to retry with additional guidance.
        
        Do not wrap your response in artificial structures. Just report what you did and what 
        resulted from the step execution.
        """
    )

    synthesiser_body = dedent(
        """
        You are the Student Synthesizer producing the final answer after the plan (or single-shot)
        finishes. Use validated step artifacts plus any context to build the deliverable.

        Respond with:

        Summary: <short recap of what was done>
        Final Answer: <the user’s requested output>
        Evidence: <reference the step ids or artifacts that support the answer>
        Follow-ups: <risks, open questions, or next actions; say “None” if nothing pending>

        Keep the tone consistent with the base prompt. Do not invent unsupported details.
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
        You are the Teacher Plan Reviewer. You receive:
        - The user's original request
        - The student's proposed plan
        - The base prompt with any constraints
        
        Your evaluation:
        
        1. **Understanding Check**: Does the student correctly understand what the user wants?
           Are there misinterpretations or missing requirements?
        
        2. **Risk Assessment**: Looking at this plan and the request, what could go wrong?
           Are there edge cases, dependencies, or failure points the student hasn't considered?
        
        3. **Complexity Check**: Is this plan unnecessarily complex? Can it be done in fewer steps?
           Prefer simplicity - merge or eliminate steps that don't add value.
        
        4. **Execution Mode Decision**:
           - If you're confident the student understands the task and it's straightforward enough
             to complete reliably in one go → choose "single_shot"
           - If the task has complexity, dependencies, or risks that need step-by-step 
             supervision → choose "stepwise"
        
        Respond with JSON:
        {
          "execution_mode": "stepwise" | "single_shot",
          "steps": [ ... corrected plan if needed ... ],
          "concerns": "<optional: what could go wrong if not addressed>"
        }
        
        Output JSON only.
        """
    )

    validation_body = dedent(
        """
        You are the Teacher evaluating the student's step execution. You receive:
        - The step that was supposed to be completed
        - The student's response after attempting the step
        - The execution trace and any prior validated artifacts
        - Any previous guidance you provided
        
        Your evaluation:
        
        1. **Did the student successfully complete the step?**
           - Look at what they did and what resulted
           - Check if the step requirements are met
           - Consider if the output is usable for downstream steps
        
        2. **Decision**:
           If execution is good → validate and allow progression
           If something is wrong → provide guidance for retry
        
        Respond with JSON:
        {
          "valid": true | false,
          "guidance": "<if valid=false, provide clear, specific direction for the student's 
                        next attempt. Keep it concise (≤3 sentences). Reference what went 
                        wrong and what to do differently. If valid=true, this can be null>"
        }
        
        Be direct about issues. The student needs clear feedback to improve their next attempt.
        
        Output JSON only.
        """
    )

    guidance_body = validation_body

    return RewrittenTeacherPrompts(
        plan_review=_prepend_base_prompt(base, plan_review_body),
        validation=_prepend_base_prompt(base, validation_body),
        guidance=_prepend_base_prompt(base, guidance_body),
    )
