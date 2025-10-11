"""Atlas SDK public entry point."""

from __future__ import annotations

import asyncio
import logging
import sys
from statistics import fmean
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
    FingerprintInputs,
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
    execution_context.metadata["persona_memories"] = {}
    execution_context.metadata["applied_persona_memories"] = {}
    execution_context.metadata["new_persona_candidates"] = []
    execution_context.metadata["persona_promotion_result"] = None
    persona_cache = get_cache()
    cache_disabled = is_cache_disabled(config)
    personas = [
        "student_planner",
        "student_executor",
        "student_synthesizer",
        "teacher_plan_review",
        "teacher_validation",
        "teacher_guidance",
    ]
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
    base_student_prompts = build_student_prompts(base_prompt, config.student)
    base_teacher_prompts = build_teacher_prompts(base_prompt, config.teacher)
    execution_context.metadata["prompt_rewrite"] = {
        "student": {
            "planner": base_student_prompts.planner,
            "executor": base_student_prompts.executor,
            "synthesizer": base_student_prompts.synthesizer,
        },
        "teacher": {
            "plan_review": base_teacher_prompts.plan_review,
            "validation": base_teacher_prompts.validation,
            "guidance": base_teacher_prompts.guidance,
        },
    }
    student = _build_student(adapter, config, base_student_prompts)
    teacher = Teacher(config.teacher, base_teacher_prompts)
    evaluator = Evaluator(config.rim)
    fingerprint_inputs: FingerprintInputs | None = None
    persona_fingerprint: str | None = None
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

        async def refresh_persona_prompts() -> None:
            nonlocal fingerprint_inputs, persona_fingerprint
            fingerprint_inputs = extract_fingerprint_inputs(task, config, execution_context)
            execution_context.metadata["persona_fingerprint_inputs"] = fingerprint_inputs
            persona_fingerprint = build_fingerprint(fingerprint_inputs)
            previous_fingerprint = execution_context.metadata.get("persona_fingerprint")
            if previous_fingerprint and previous_fingerprint != persona_fingerprint:
                for persona_id in personas:
                    persona_cache.invalidate(
                        PersonaMemoryKey(
                            agent_name=fingerprint_inputs.agent_name,
                            tenant_id=fingerprint_inputs.tenant_id,
                            fingerprint=previous_fingerprint,
                            persona=persona_id,
                        )
                    )
            execution_context.metadata["persona_fingerprint"] = persona_fingerprint
            persona_memories = execution_context.metadata.setdefault("persona_memories", {})
            persona_memories.clear()
            applied_memories = execution_context.metadata.setdefault("applied_persona_memories", {})
            applied_memories.clear()
            instructions_map: Dict[str, List[Any]] = {}
            trial_instructions_map: Dict[str, List[Any]] = {}
            if database and persona_fingerprint:
                statuses = ["active", "candidate"]
                use_cache = not cache_disabled
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
                    normalized = normalize_instructions(records)
                    if normalized:
                        actives: List[Any] = []
                        trials: List[Any] = []
                        for inst in normalized:
                            if inst.status == "candidate":
                                trials.append(inst)
                            else:
                                actives.append(inst)
                        if actives:
                            instructions_map[persona_id] = actives
                        if trials:
                            trial_instructions_map[persona_id] = trials
            else:
                execution_context.metadata["persona_memories"] = persona_memories

            def _apply_instructions(persona_id: str, prompt_text: str) -> str:
                instructions = (instructions_map.get(persona_id) or []) + (trial_instructions_map.get(persona_id) or [])
                if not instructions:
                    return prompt_text
                applied_entries: List[Dict[str, Any]] = []
                seen: Dict[Any, None] = {}
                for inst in instructions:
                    identifier = inst.memory_id
                    if identifier is None or identifier in seen:
                        continue
                    seen[identifier] = None
                    applied_entries.append(
                        {
                            "memory_id": identifier,
                            "status": inst.status or "active",
                        }
                    )
                applied_memories[persona_id] = applied_entries
                return merge_prompt(prompt_text, instructions)

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
                "student": {
                    "planner": student_prompts.planner,
                    "executor": student_prompts.executor,
                    "synthesizer": student_prompts.synthesizer,
                },
                "teacher": {
                    "plan_review": teacher_prompts.plan_review,
                    "validation": teacher_prompts.validation,
                    "guidance": teacher_prompts.guidance,
                },
            }
            student.update_prompts(student_prompts)
            teacher.update_prompts(teacher_prompts)

        orchestrator = Orchestrator(
            teacher=teacher,
            student=student,
            evaluator=evaluator,
            orchestration_config=config.orchestration,
            rim_config=config.rim,
            persona_refresh=refresh_persona_prompts,
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
                if fingerprint_inputs and persona_fingerprint:
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
                else:
                    promotion_payload = None
            except Exception as promotion_exc:  # pragma: no cover - diagnostic path
                logger.exception("Failed to promote persona memories", exc_info=promotion_exc)
            finally:
                execution_context.metadata["persona_promotion_result"] = promotion_payload
            await _log_persona_memory_usage(database, session_id, execution_context, result)
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
            await _log_persona_memory_usage(database, session_id, execution_context, None)
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


async def _log_persona_memory_usage(
    database: Database,
    session_id: int,
    context: ExecutionContext,
    result: Result | None,
) -> None:
    applied = context.metadata.get("applied_persona_memories") or {}
    if not applied:
        return
    usage_metrics = _collect_persona_usage_metrics(context, result)
    default_metric = usage_metrics.get("__default__")
    logged: set[Any] = set()
    for persona_id, entries in applied.items():
        metric = usage_metrics.get(persona_id, default_metric) if usage_metrics else default_metric
        reward_payload = metric["reward"] if metric else None
        retry_count = metric["retries"] if metric else None
        normalised_ids = []
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict):
                    candidate_id = entry.get("memory_id")
                    if candidate_id is not None:
                        normalised_ids.append(candidate_id)
                else:
                    normalised_ids.append(entry)
        else:
            normalised_ids = entries
        for memory_id in normalised_ids:
            if memory_id and memory_id not in logged:
                await database.log_persona_memory_usage(
                    memory_id,
                    session_id,
                    reward=reward_payload,
                    retries=retry_count,
                )
                logged.add(memory_id)


def _collect_persona_usage_metrics(context: ExecutionContext, result: Result | None) -> dict[str, dict[str, Any]]:
    if result is None:
        return {}
    steps_metadata = context.metadata.get("steps", {}) or {}
    persona_scores: dict[str, list[float]] = {}
    persona_attempts: dict[str, list[int]] = {}
    all_scores: list[float] = []
    all_attempts: list[int] = []

    for step_result in result.step_results:
        step_meta = steps_metadata.get(step_result.step_id, {}) or {}
        persona_id = _infer_persona_from_step(step_result, step_meta)
        score = _extract_reward_score(step_result, step_meta)
        attempts = step_result.attempts or len(step_meta.get("attempts") or []) or 0

        if score is not None:
            persona_scores.setdefault(persona_id, []).append(score)
            all_scores.append(score)
        if attempts:
            persona_attempts.setdefault(persona_id, []).append(int(attempts))
            all_attempts.append(int(attempts))

    usage_metrics: dict[str, dict[str, Any]] = {}

    default_reward = fmean(all_scores) if all_scores else None
    default_attempts = round(fmean(all_attempts)) if all_attempts else None

    for persona_id in set(list(persona_scores.keys()) + list(persona_attempts.keys())):
        rewards = persona_scores.get(persona_id) or []
        attempts = persona_attempts.get(persona_id) or []
        persona_reward = fmean(rewards) if rewards else default_reward
        persona_attempt = round(fmean(attempts)) if attempts else default_attempts
        usage_metrics[persona_id] = {
            "reward": {"score": float(persona_reward)} if persona_reward is not None else None,
            "retries": persona_attempt,
        }

    usage_metrics["__default__"] = {
        "reward": {"score": float(default_reward)} if default_reward is not None else None,
        "retries": default_attempts,
    }
    return usage_metrics


def _infer_persona_from_step(step_result, step_metadata: dict[str, Any]) -> str:
    metadata = step_result.metadata or {}
    persona = metadata.get("persona_target") or metadata.get("persona") or metadata.get("actor")
    if isinstance(persona, str) and persona:
        return persona
    guidance_source = step_metadata.get("guidance_source")
    if isinstance(guidance_source, str) and guidance_source:
        return guidance_source
    return "student_executor"


def _extract_reward_score(step_result, step_metadata: dict[str, Any]) -> float | None:
    evaluation = getattr(step_result, "evaluation", None)
    reward = getattr(evaluation, "reward", None) if evaluation is not None else None
    if reward is not None:
        score = getattr(reward, "score", None)
        if isinstance(score, (int, float)):
            return float(score)
        to_dict = getattr(reward, "to_dict", None)
        if callable(to_dict):
            reward_dict = to_dict()
            score = reward_dict.get("score") if isinstance(reward_dict, dict) else None
            if isinstance(score, (int, float)):
                return float(score)
    attempts_meta = step_metadata.get("attempts") or []
    for attempt in reversed(attempts_meta):
        if not isinstance(attempt, dict):
            continue
        evaluation_payload = attempt.get("evaluation")
        if not isinstance(evaluation_payload, dict):
            continue
        reward_payload = evaluation_payload.get("reward")
        if isinstance(reward_payload, dict):
            score = reward_payload.get("score")
            if isinstance(score, (int, float)):
                return float(score)
    return None
