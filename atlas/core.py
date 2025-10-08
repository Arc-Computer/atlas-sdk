"""Atlas SDK public entry point."""

from __future__ import annotations

import asyncio
import sys
from textwrap import dedent
from typing import Any
from typing import List
from typing import Protocol

from atlas.agent.factory import create_from_atlas_config
from atlas.config.loader import load_config
from atlas.config.models import AtlasConfig
from atlas.config.models import StudentConfig, TeacherConfig
from atlas.orchestration.execution_context import ExecutionContext
from atlas.orchestration.orchestrator import Orchestrator
from atlas.reward.evaluator import Evaluator
from atlas.roles.student import Student
from atlas.roles.teacher import Teacher
from atlas.storage.database import Database
from atlas.telemetry import ConsoleTelemetryStreamer
from atlas.telemetry.langchain_callback import configure_langchain_callbacks
from atlas.transition.rewriter import (
    RewrittenStudentPrompts,
    RewrittenTeacherPrompts,
)
from atlas.types import Result


class TelemetryPublisherProtocol(Protocol):
    def attach(self, step_manager: Any) -> None:
        ...

    def detach(self) -> None:
        ...

    def publish_control_event(self, event_type: str, data: dict[str, Any]) -> None:
        ...


def _prepend_base_prompt(base_prompt: str, body: str) -> str:
    base = base_prompt.strip()
    if base:
        return f"{base}\n\n{body.strip()}"
    return body.strip()


def _format_with_base(template: str, base_prompt: str) -> str:
    if "{base_prompt}" in template:
        return template.replace("{base_prompt}", base_prompt.strip())
    return template


def _default_student_prompts(base_prompt: str, student_cfg: StudentConfig) -> RewrittenStudentPrompts:
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


def _default_teacher_prompts(base_prompt: str, teacher_cfg: TeacherConfig) -> RewrittenTeacherPrompts:
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

async def arun(
    task: str,
    config_path: str,
    publisher: TelemetryPublisherProtocol | None = None,
    session_metadata: dict[str, Any] | None = None,
    stream_progress: bool | None = None,
) -> Result:
    config = load_config(config_path)
    execution_context = ExecutionContext.get()
    execution_context.reset()
    configure_langchain_callbacks()
    if session_metadata:
        execution_context.metadata["session_metadata"] = session_metadata
    if stream_progress is not None:
        stream_enabled = stream_progress
    else:
        isatty = getattr(sys.stdout, "isatty", None)
        stream_enabled = bool(isatty and isatty())
    streamer: ConsoleTelemetryStreamer | None = None
    events: List = []
    subscription = execution_context.event_stream.subscribe(events.append)
    if publisher is not None:
        publisher.attach(execution_context.intermediate_step_manager)
    elif stream_enabled:
        streamer = ConsoleTelemetryStreamer()
        streamer.attach(execution_context)
        streamer.session_started(task)
    adapter = create_from_atlas_config(config)
    adapter_config = config.agent
    base_prompt = getattr(adapter_config, "system_prompt", "")
    if config.prompt_rewrite is not None:
        raise ValueError(
            "prompt_rewrite configuration is no longer supported. Remove the prompt_rewrite block "
            "from your Atlas config and rely on explicit student/teacher prompts."
        )
    student_prompts = _default_student_prompts(base_prompt, config.student)
    teacher_prompts = _default_teacher_prompts(base_prompt, config.teacher)
    execution_context.metadata["prompt_rewrite"] = {
        "student": student_prompts.__dict__,
        "teacher": teacher_prompts.__dict__,
    }
    student = _build_student(adapter, config, student_prompts)
    teacher = Teacher(config.teacher, teacher_prompts)
    evaluator = Evaluator(config.rim)
    orchestrator = Orchestrator(
        teacher=teacher,
        student=student,
        evaluator=evaluator,
        orchestration_config=config.orchestration,
        rim_config=config.rim,
    )
    database = Database(config.storage) if config.storage else None
    session_id: int | None = None
    try:
        if database:
            await database.connect()
            metadata = execution_context.metadata.get("session_metadata")
            session_id = await database.create_session(task, metadata=metadata)
            if publisher is not None and session_id is not None:
                publisher.publish_control_event(
                    "session-started",
                    {"session_id": session_id, "task": task},
                )
        result = await orchestrator.arun(task)
        if database and session_id is not None:
            await _persist_results(database, session_id, execution_context, result, events)
            await database.finalize_session(session_id, result.final_answer, "succeeded")
            if publisher is not None:
                publisher.publish_control_event(
                    "session-completed",
                    {
                        "session_id": session_id,
                        "status": "succeeded",
                        "final_answer": result.final_answer,
                    },
                )
        if streamer is not None:
            streamer.session_completed(result)
        return result
    except Exception as exc:
        if database and session_id is not None:
            await _persist_events(database, session_id, events)
            await database.finalize_session(session_id, "", "failed")
            if publisher is not None:
                publisher.publish_control_event(
                    "session-completed",
                    {"session_id": session_id, "status": "failed"},
                )
        if streamer is not None:
            streamer.session_failed(exc)
        raise
    finally:
        subscription.unsubscribe()
        if publisher is not None:
            publisher.detach()
        elif streamer is not None:
            streamer.detach()
        if database:
            await database.disconnect()




def run(
    task: str,
    config_path: str,
    publisher: TelemetryPublisherProtocol | None = None,
    session_metadata: dict[str, Any] | None = None,
    stream_progress: bool | None = None,
) -> Result:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            arun(
                task,
                config_path,
                publisher=publisher,
                session_metadata=session_metadata,
                stream_progress=stream_progress,
            )
        )
    raise RuntimeError("atlas.run cannot be invoked inside an existing event loop")


def _build_student(adapter, config: AtlasConfig, student_prompts) -> Student:
    adapter_config = config.agent
    return Student(
        adapter=adapter,
        adapter_config=adapter_config,
        student_config=config.student,
        student_prompts=student_prompts,
    )


async def _persist_results(
    database: Database,
    session_id: int,
    context: ExecutionContext,
    result: Result,
    events: List,
) -> None:
    await database.log_plan(session_id, result.plan)
    steps_metadata = context.metadata.get("steps", {})
    for step_result in result.step_results:
        await database.log_step_result(session_id, step_result)
        step_meta = steps_metadata.get(step_result.step_id, {})
        await database.log_step_attempts(session_id, step_result.step_id, step_meta.get("attempts", []))
        await database.log_guidance(session_id, step_result.step_id, step_meta.get("guidance", []))
    await _persist_events(database, session_id, events)


async def _persist_events(database: Database, session_id: int, events: List) -> None:
    for event in events:
        await database.log_intermediate_step(session_id, event)
