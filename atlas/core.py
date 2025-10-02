"""Atlas SDK public entry point."""

from __future__ import annotations

import asyncio
from typing import List

from atlas.agent.factory import create_from_atlas_config
from atlas.config.loader import load_config
from atlas.config.models import AtlasConfig
from atlas.orchestration.execution_context import ExecutionContext
from atlas.orchestration.orchestrator import Orchestrator
from atlas.reward.evaluator import Evaluator
from atlas.roles.student import Student
from atlas.roles.teacher import Teacher
from atlas.storage.database import Database
from atlas.transition.rewriter import (
    PromptRewriteEngine,
    RewrittenStudentPrompts,
    RewrittenTeacherPrompts,
)
from atlas.types import Result


async def arun(task: str, config_path: str) -> Result:
    config = load_config(config_path)
    execution_context = ExecutionContext.get()
    execution_context.reset()
    events: List = []
    subscription = execution_context.event_stream.subscribe(events.append)
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
            session_id = await database.create_session(task)
        result = await orchestrator.arun(task)
        if database and session_id is not None:
            await _persist_results(database, session_id, execution_context, result, events)
            await database.finalize_session(session_id, result.final_answer, "succeeded")
        return result
    except Exception:
        if database and session_id is not None:
            await _persist_events(database, session_id, events)
            await database.finalize_session(session_id, "", "failed")
        raise
    finally:
        subscription.unsubscribe()
        if database:
            await database.disconnect()




def run(task: str, config_path: str) -> Result:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(arun(task, config_path))
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
