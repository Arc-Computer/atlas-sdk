"""Atlas SDK public entry point."""

from __future__ import annotations

import asyncio
import sys
from typing import Any
from typing import List
from typing import Protocol

from atlas.agent.factory import create_from_atlas_config
from atlas.config.loader import load_config
from atlas.config.models import AtlasConfig
from atlas.orchestration.execution_context import ExecutionContext
from atlas.orchestration.orchestrator import Orchestrator
from atlas.reward.evaluator import Evaluator
from atlas.roles.student import Student
from atlas.roles.teacher import Teacher
from atlas.storage.database import Database
from atlas.telemetry import ConsoleTelemetryStreamer
from atlas.telemetry.langchain_callback import configure_langchain_callbacks
from atlas.transition.rewriter import (
    PromptRewriteEngine,
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
    rewrite_engine = PromptRewriteEngine(config.prompt_rewrite, getattr(adapter_config, "llm", None))
    student_prompts, teacher_prompts = await rewrite_engine.generate(
        base_prompt=getattr(adapter_config, "system_prompt", ""),
        adapter_config=adapter_config,
        student_config=config.student,
        teacher_config=config.teacher,
    )
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
