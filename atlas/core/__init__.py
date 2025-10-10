"""Atlas SDK public entry point."""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any
from typing import Dict
from typing import List
from typing import Protocol

from atlas.connectors.factory import create_from_atlas_config
from atlas.config.loader import load_config
from atlas.config.models import AtlasConfig
from atlas.prompts import (
    RewrittenStudentPrompts,
    RewrittenTeacherPrompts,
    build_student_prompts,
    build_teacher_prompts,
)
from atlas.runtime.orchestration.execution_context import ExecutionContext
from atlas.runtime.orchestration.orchestrator import Orchestrator
from atlas.runtime.persona_memory import (
    PersonaMemoryKey,
    build_fingerprint,
    extract_candidates,
    extract_fingerprint_inputs,
    get_cache,
    get_promotion_settings,
    is_cache_disabled,
    merge_prompt,
    normalize_instructions,
    promote_and_compact,
    write_candidates,
)
from atlas.evaluation.evaluator import Evaluator
from atlas.personas.student import Student
from atlas.personas.teacher import Teacher
from atlas.runtime.storage.database import Database
from atlas.runtime.telemetry import ConsoleTelemetryStreamer
from atlas.runtime.telemetry.langchain_callback import configure_langchain_callbacks
from atlas.types import Result

logger = logging.getLogger(__name__)


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
    else:
        execution_context.metadata.setdefault("session_metadata", {})
    fingerprint_inputs = extract_fingerprint_inputs(task, config, execution_context)
    execution_context.metadata["persona_fingerprint_inputs"] = fingerprint_inputs
    persona_fingerprint = build_fingerprint(fingerprint_inputs)
    execution_context.metadata["persona_fingerprint"] = persona_fingerprint
    execution_context.metadata["persona_memories"] = {}
    execution_context.metadata["applied_persona_memories"] = {}
    execution_context.metadata["new_persona_candidates"] = []
    execution_context.metadata["persona_promotion_result"] = None
    persona_cache = get_cache()
    cache_disabled = is_cache_disabled(config)
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
    database = Database(config.storage) if config.storage else None
    session_id: int | None = None
    try:
        persona_memories = execution_context.metadata["persona_memories"]
        if database:
            await database.connect()
            metadata = execution_context.metadata.get("session_metadata")
            session_id = await database.create_session(task, metadata=metadata)
            if persona_fingerprint:
                persona_memories = execution_context.metadata.setdefault("persona_memories", {})
                statuses = ["active"]
                use_cache = not cache_disabled
                personas = [
                    "student_planner",
                    "student_executor",
                    "student_synthesizer",
                    "teacher_plan_review",
                    "teacher_validation",
                    "teacher_guidance",
                ]
                for persona_id in personas:
                    key = PersonaMemoryKey(
                        agent_name=fingerprint_inputs.agent_name,
                        tenant_id=fingerprint_inputs.tenant_id,
                        fingerprint=persona_fingerprint,
                        persona=persona_id,
                    )
                    records = await persona_cache.get_or_load(
                        database,
                        key,
                        statuses,
                        use_cache=use_cache,
                    )
                    persona_memories[persona_id] = records
            if publisher is not None and session_id is not None:
                publisher.publish_control_event(
                    "session-started",
                    {"session_id": session_id, "task": task},
                )
        instructions_map = {persona: normalize_instructions(records) for persona, records in persona_memories.items() if records}
        applied_memories = execution_context.metadata["applied_persona_memories"]

        def _apply_instructions(persona_id: str, prompt_text: str) -> str:
            instructions = instructions_map.get(persona_id) or []
            if not instructions:
                return prompt_text
            ids = [inst.memory_id for inst in instructions if inst.memory_id is not None]
            applied_memories[persona_id] = list(dict.fromkeys(ids))
            return merge_prompt(prompt_text, instructions)

        base_student_prompts = build_student_prompts(base_prompt, config.student)
        base_teacher_prompts = build_teacher_prompts(base_prompt, config.teacher)
        student_prompts = RewrittenStudentPrompts(
            planner=_apply_instructions("student_planner", base_student_prompts.planner),
            executor=_apply_instructions("student_executor", base_student_prompts.executor),
            synthesizer=_apply_instructions("student_synthesizer", base_student_prompts.synthesizer),
        )
        teacher_prompts = RewrittenTeacherPrompts(
            plan_review=_apply_instructions("teacher_plan_review", base_teacher_prompts.plan_review),
            validation=_apply_instructions("teacher_validation", base_teacher_prompts.validation),
            guidance=_apply_instructions("teacher_guidance", base_teacher_prompts.guidance),
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
        result = await orchestrator.arun(task)
        if database and session_id is not None:
            await _persist_results(database, session_id, execution_context, result, events)
            candidate_ids: List[str] = []
            try:
                candidate_specs = extract_candidates(execution_context, result)
                if candidate_specs:
                    created = await write_candidates(database, session_id, candidate_specs)
                    candidate_ids = [str(identifier) for identifier in created]
            except Exception as learning_exc:  # pragma: no cover - diagnostic path
                logger.exception("Failed to generate persona memory candidates", exc_info=learning_exc)
            finally:
                execution_context.metadata["new_persona_candidates"] = candidate_ids
            promotion_payload: Dict[str, Any] | None = None
            try:
                promotion_settings = get_promotion_settings(config)
                promotion_result = await promote_and_compact(
                    database,
                    fingerprint_inputs,
                    persona_fingerprint,
                    promotion_settings,
                )
                if promotion_result.invalidate_personas:
                    for persona_id in promotion_result.invalidate_personas:
                        key = PersonaMemoryKey(
                            agent_name=fingerprint_inputs.agent_name,
                            tenant_id=fingerprint_inputs.tenant_id,
                            fingerprint=persona_fingerprint,
                            persona=persona_id,
                        )
                        persona_cache.invalidate(key)
                promotion_payload = promotion_result.to_dict()
            except Exception as promotion_exc:  # pragma: no cover - diagnostic path
                logger.exception("Failed to promote persona memories", exc_info=promotion_exc)
            finally:
                execution_context.metadata["persona_promotion_result"] = promotion_payload
            await _log_persona_memory_usage(database, session_id, execution_context)
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
            await _log_persona_memory_usage(database, session_id, execution_context)
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


async def _log_persona_memory_usage(database: Database, session_id: int, context: ExecutionContext) -> None:
    applied = context.metadata.get("applied_persona_memories") or {}
    if not applied:
        return
    logged: set[Any] = set()
    for memory_ids in applied.values():
        for memory_id in memory_ids:
            if memory_id and memory_id not in logged:
                await database.log_persona_memory_usage(memory_id, session_id, reward=None, retries=None)
                logged.add(memory_id)
