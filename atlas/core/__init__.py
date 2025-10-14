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
from datetime import datetime, timezone
from importlib import import_module

from atlas.connectors.factory import create_from_atlas_config
from atlas.config.loader import load_config
from atlas.config.models import AdaptiveTeachingConfig, AtlasConfig, RewardObjectiveConfig
from atlas.prompts import (
    RewrittenStudentPrompts,
    RewrittenTeacherPrompts,
    build_student_prompts,
    build_teacher_prompts,
)
from atlas.runtime.orchestration.execution_context import ExecutionContext
from atlas.runtime.adaptive import CapabilityProbeClient
from atlas.runtime.orchestration.orchestrator import Orchestrator
from atlas.runtime.persona_memory import (
    CANONICAL_PERSONAS,
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
    canonical_persona_name,
    persona_aliases,
    write_candidates,
)
from atlas.evaluation.evaluator import Evaluator
from atlas.personas.student import Student
from atlas.personas.teacher import Teacher
from atlas.runtime.storage.database import Database
from atlas.runtime.telemetry import ConsoleTelemetryStreamer
from atlas.runtime.telemetry.langchain_callback import configure_langchain_callbacks
from atlas.types import Result
from atlas.utils.triage import default_build_dossier

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
    personas = list(CANONICAL_PERSONAS)
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
    adaptive_teaching_cfg = getattr(config, "adaptive_teaching", AdaptiveTeachingConfig())
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
    teacher = Teacher(config.teacher, base_teacher_prompts, adapter_config.tools)
    evaluator = _build_evaluator_instance(config, getattr(adaptive_teaching_cfg, "reward", None))
    execution_context.metadata["adaptive_default_tags"] = list(getattr(adaptive_teaching_cfg, "default_tags", []) or [])
    triage_adapter = _load_triage_adapter(getattr(adaptive_teaching_cfg, "triage_adapter", None))
    fingerprint_inputs: FingerprintInputs | None = extract_fingerprint_inputs(task, config, execution_context)
    persona_fingerprint: str | None = build_fingerprint(fingerprint_inputs)
    execution_context.metadata["persona_fingerprint_inputs"] = fingerprint_inputs
    execution_context.metadata["persona_fingerprint"] = persona_fingerprint
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
                    for alias in persona_aliases(persona_id):
                        persona_cache.invalidate(
                            PersonaMemoryKey(
                                agent_name=fingerprint_inputs.agent_name,
                                tenant_id=fingerprint_inputs.tenant_id,
                                fingerprint=previous_fingerprint,
                                persona=alias,
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
                    combined_records: List[Any] = []
                    for alias in persona_aliases(persona_id):
                        key = PersonaMemoryKey(
                            agent_name=fingerprint_inputs.agent_name,
                            tenant_id=fingerprint_inputs.tenant_id,
                            fingerprint=persona_fingerprint,
                            persona=alias,
                        )
                        records = await persona_cache.get_or_load(
                            database,
                            key,
                            statuses,
                            use_cache=use_cache,
                        )
                        if records:
                            combined_records.extend(records)
                        persona_memories[alias] = records or []
                    persona_memories[persona_id] = combined_records
                    normalized = normalize_instructions(combined_records)
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
                for alias in persona_aliases(persona_id):
                    applied_memories[alias] = list(applied_entries)
                return merge_prompt(prompt_text, instructions)

            student_prompts = RewrittenStudentPrompts(
                planner=_apply_instructions("student", base_student_prompts.planner),
                executor=_apply_instructions("student", base_student_prompts.executor),
                synthesizer=_apply_instructions("student", base_student_prompts.synthesizer),
            )
            teacher_prompts = RewrittenTeacherPrompts(
                plan_review=_apply_instructions("teacher_plan_review", base_teacher_prompts.plan_review),
                validation=_apply_instructions("teacher_validation", base_teacher_prompts.validation),
                guidance=_apply_instructions("teacher_validation", base_teacher_prompts.guidance),
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

        capability_probe_client = CapabilityProbeClient(adaptive_teaching_cfg.probe)

        orchestrator = Orchestrator(
            teacher=teacher,
            student=student,
            evaluator=evaluator,
            orchestration_config=config.orchestration,
            rim_config=config.rim,
            adaptive_config=adaptive_teaching_cfg,
            triage_adapter=triage_adapter,
            persona_refresh=refresh_persona_prompts,
            capability_probe=capability_probe_client,
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

                    adaptive_meta = execution_context.metadata.get("adaptive", {})
                    was_certification_run = adaptive_meta.get("certification_run") if isinstance(adaptive_meta, dict) else False

                    if was_certification_run and created and persona_fingerprint:
                        active_mode = adaptive_meta.get("active_mode") if isinstance(adaptive_meta, dict) else "paired"
                        for memory_uuid in created:
                            try:
                                updated_metadata = _update_persona_metadata_payload(
                                    {},
                                    None,
                                    active_mode,
                                    fingerprint=persona_fingerprint,
                                    certified=True,
                                )
                                if updated_metadata:
                                    await database.update_persona_metadata(memory_uuid, updated_metadata)
                            except Exception as cert_exc:
                                logger.exception("Failed to mark persona memory as certified: %s", cert_exc)
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
                            for alias in persona_aliases(persona_id):
                                key = PersonaMemoryKey(
                                    agent_name=fingerprint_inputs.agent_name,
                                    tenant_id=fingerprint_inputs.tenant_id,
                                    fingerprint=persona_fingerprint,
                                    persona=alias,
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


def _build_evaluator_instance(
    config: AtlasConfig,
    reward_cfg: RewardObjectiveConfig | None,
):
    reward_cfg = reward_cfg or RewardObjectiveConfig()
    if reward_cfg.type == "rim":
        rim_config = config.rim
        if reward_cfg.parameters:
            try:
                rim_config = rim_config.model_copy(update=reward_cfg.parameters)
            except Exception as exc:  # pragma: no cover - defensive guard
                raise ValueError(f"Invalid adaptive_teaching.reward.parameters: {exc}") from exc
        focus_prompt = reward_cfg.focus_prompt or getattr(rim_config, "judge_prompt", None)
        return Evaluator(rim_config, focus_prompt=focus_prompt)
    if reward_cfg.type == "python":
        if not reward_cfg.import_path and not reward_cfg.attribute:
            raise ValueError("adaptive_teaching.reward.import_path is required when type='python'")
        factory = _resolve_callable(reward_cfg.import_path, reward_cfg.attribute)
        try:
            evaluator = factory(config=config, reward_config=reward_cfg)
        except TypeError:
            evaluator = factory(config, reward_cfg)
        if evaluator is None:
            raise ValueError("adaptive_teaching.reward factory returned None")
        return evaluator
    raise ValueError(f"Unsupported reward type: {reward_cfg.type}")


def _load_triage_adapter(path: str | None):
    if not path:
        return default_build_dossier
    try:
        adapter = _resolve_callable(path, None)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.warning("Falling back to default triage adapter due to error: %s", exc)
        return default_build_dossier
    if not callable(adapter):
        logger.warning("Triage adapter %s is not callable; using default adapter instead", path)
        return default_build_dossier
    return adapter


def _split_callable_path(path: str) -> tuple[str, str]:
    if ":" in path:
        module_path, attribute = path.split(":", 1)
    elif "." in path:
        module_path, attribute = path.rsplit(".", 1)
    else:
        raise ValueError(f"Invalid adapter path '{path}'. Expected 'module:callable' or 'module.callable'.")
    module_path = module_path.strip()
    attribute = attribute.strip()
    if not module_path or not attribute:
        raise ValueError(f"Invalid adapter path '{path}'.")
    return module_path, attribute


def _resolve_callable(path: str | None, attribute: str | None):
    if path and attribute:
        module = import_module(path)
        return getattr(module, attribute)
    if not path:
        raise ValueError("import path must be provided")
    module_path, attr = _split_callable_path(path)
    module = import_module(module_path)
    return getattr(module, attr)


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
    session_reward = context.metadata.get("session_reward")
    student_learning = context.metadata.get("session_student_learning")
    teacher_learning = context.metadata.get("session_teacher_learning")
    if session_reward is not None or student_learning is not None or teacher_learning is not None:
        await database.log_session_reward(
            session_id,
            session_reward,
            student_learning,
            teacher_learning,
        )
    await _update_session_metadata(database, session_id, context, result)
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
    adaptive_meta = context.metadata.get("adaptive", {}) if isinstance(context.metadata, dict) else {}
    persona_records = context.metadata.get("persona_memories") if isinstance(context.metadata, dict) else {}
    persona_lookup: dict[Any, dict[str, Any]] = {}
    if isinstance(persona_records, dict):
        for records in persona_records.values():
            if isinstance(records, list):
                for record in records:
                    memory_id = record.get("memory_id")
                    if memory_id is not None:
                        persona_lookup[memory_id] = record
                        persona_lookup[str(memory_id)] = record
    certified_store = context.metadata.get("adaptive_certified_fingerprints")
    if isinstance(certified_store, set):
        certified_set = certified_store
    elif isinstance(certified_store, list):
        certified_set = set(str(item) for item in certified_store)
    else:
        certified_set = set()
    for persona_id, entries in applied.items():
        canonical_persona = canonical_persona_name(persona_id)
        metric = usage_metrics.get(canonical_persona, default_metric) if usage_metrics else default_metric
        reward_payload = metric["reward"] if metric else None
        retry_count = metric["retries"] if metric else None
        mode_used = metric.get("mode") if metric else None
        if mode_used is None and isinstance(adaptive_meta, dict):
            mode_used = adaptive_meta.get("active_mode")
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
                    mode=mode_used,
                )
                record = persona_lookup.get(memory_id)
                metadata_record = record.get("metadata") if isinstance(record, dict) else None
                fingerprint = record.get("trigger_fingerprint") if isinstance(record, dict) else None
                certified_flag = isinstance(fingerprint, str) and fingerprint in certified_set
                updated_metadata = _update_persona_metadata_payload(
                    metadata_record if isinstance(metadata_record, dict) else {},
                    reward_payload,
                    mode_used,
                    fingerprint=fingerprint if isinstance(fingerprint, str) else None,
                    certified=certified_flag,
                )
                if updated_metadata is not None:
                    await database.update_persona_metadata(memory_id, updated_metadata)
                    if persona_lookup.get(memory_id) is not None:
                        persona_lookup[memory_id]["metadata"] = updated_metadata
                logged.add(memory_id)


def _collect_persona_usage_metrics(context: ExecutionContext, result: Result | None) -> dict[str, dict[str, Any]]:
    if result is None:
        return {}
    steps_metadata = context.metadata.get("steps", {}) or {}
    adaptive_meta = context.metadata.get("adaptive", {}) if isinstance(context.metadata, dict) else {}
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
            "mode": adaptive_meta.get("active_mode") if isinstance(adaptive_meta, dict) else None,
        }

    usage_metrics["__default__"] = {
        "reward": {"score": float(default_reward)} if default_reward is not None else None,
        "retries": default_attempts,
        "mode": adaptive_meta.get("active_mode") if isinstance(adaptive_meta, dict) else None,
    }
    return usage_metrics


def _update_persona_metadata_payload(
    original: Dict[str, Any],
    reward_payload: Dict[str, Any] | None,
    mode: str | None,
    *,
    fingerprint: str | None,
    certified: bool,
) -> Dict[str, Any] | None:
    metadata = dict(original or {})
    tags: set[str] = set()
    existing_tags = metadata.get("tags")
    if isinstance(existing_tags, list):
        tags.update(tag for tag in existing_tags if isinstance(tag, str))
    metadata.setdefault("helpful_count", 0)
    metadata.setdefault("harmful_count", 0)
    metadata.setdefault("neutral_count", 0)
    score: float | None = None
    if isinstance(reward_payload, dict):
        value = reward_payload.get("score")
        if isinstance(value, (int, float)):
            raw_payload = reward_payload.get("raw")
            if isinstance(raw_payload, dict) and raw_payload.get("skipped"):
                score = None
            else:
                score = float(value)
    classification = None
    if score is not None:
        classification = _classify_reward(score)
        if classification == "helpful":
            metadata["helpful_count"] = int(metadata.get("helpful_count", 0) or 0) + 1
        elif classification == "harmful":
            metadata["harmful_count"] = int(metadata.get("harmful_count", 0) or 0) + 1
        else:
            metadata["neutral_count"] = int(metadata.get("neutral_count", 0) or 0) + 1
        metadata["last_reward"] = score
        metadata["last_reward_at"] = datetime.now(timezone.utc).isoformat()
    if isinstance(mode, str) and mode:
        metadata["last_mode"] = mode
        tags.add(f"last_mode:{mode}")
        if classification == "helpful" and mode == "auto":
            tags.add("auto_mode_success")
        if classification == "harmful" and mode == "escalate":
            tags.add("escalate_intervention")
    if certified:
        metadata["certified"] = True
        tags.add("certified")
        if fingerprint:
            tags.add(f"fingerprint:{fingerprint}")
    metadata["tags"] = sorted(tags)
    return metadata


def _classify_reward(score: float) -> str:
    if score >= 0.8:
        return "helpful"
    if score <= 0.3:
        return "harmful"
    return "neutral"


async def _update_session_metadata(
    database: Database,
    session_id: int,
    context: ExecutionContext,
    result: Result,
) -> None:
    base_metadata = context.metadata.get("session_metadata") or {}
    if not isinstance(base_metadata, dict):
        base_metadata = {}
    insights = _collect_session_insights(context, result)
    if not insights:
        return
    merged = {**base_metadata, **insights}
    context.metadata["session_metadata"] = merged
    await database.update_session_metadata(session_id, merged)


def _collect_session_insights(context: ExecutionContext, result: Result) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    triage = context.metadata.get("triage", {}).get("dossier") if isinstance(context.metadata, dict) else None
    if triage:
        payload["triage_dossier"] = triage
    adaptive_summary = _collect_adaptive_summary(context)
    if adaptive_summary:
        context.metadata["adaptive_summary"] = adaptive_summary
        payload["adaptive_summary"] = adaptive_summary
    session_reward = context.metadata.get("session_reward") if isinstance(context.metadata, dict) else None
    if session_reward is not None:
        reward_payload = session_reward.to_dict() if hasattr(session_reward, "to_dict") else session_reward
        payload["session_reward"] = reward_payload
    student_learning = context.metadata.get("session_student_learning") if isinstance(context.metadata, dict) else None
    if isinstance(student_learning, str) and student_learning.strip():
        payload["student_learning"] = student_learning
    teacher_learning = context.metadata.get("session_teacher_learning") if isinstance(context.metadata, dict) else None
    if isinstance(teacher_learning, str) and teacher_learning.strip():
        payload["teacher_learning"] = teacher_learning
    session_reward_payload = payload.get("session_reward")
    if session_reward_payload:
        raw_score = None
        if isinstance(session_reward_payload, dict):
            raw_score = session_reward_payload.get("score")
        payload["reward_summary"] = {"score": raw_score}
    else:
        payload["reward_summary"] = _collect_reward_summary(result)
    personas_used = _collect_personas_used(context)
    if personas_used:
        payload["personas_used"] = personas_used
    persona_updates = _collect_persona_updates(context)
    if persona_updates:
        payload["persona_updates"] = persona_updates
    teacher_notes = _extract_teacher_notes(context)
    if teacher_notes:
        payload["teacher_notes"] = teacher_notes
    return payload


def _collect_adaptive_summary(context: ExecutionContext) -> dict[str, Any]:
    adaptive_meta = context.metadata.get("adaptive") if isinstance(context.metadata, dict) else None
    if not isinstance(adaptive_meta, dict):
        return {}
    summary: dict[str, Any] = {}
    mode = adaptive_meta.get("active_mode")
    if isinstance(mode, str):
        summary["adaptive_mode"] = mode
    history = adaptive_meta.get("mode_history")
    if isinstance(history, list) and history:
        summary["mode_history"] = history
        last_entry = history[-1]
        if isinstance(last_entry, dict) and last_entry.get("confidence") is not None:
            summary["confidence"] = last_entry.get("confidence")
    summary["certification_run"] = bool(adaptive_meta.get("certification_run"))
    probe_payload = adaptive_meta.get("probe")
    if isinstance(probe_payload, dict):
        summary["probe"] = probe_payload
    return summary


def _collect_personas_used(context: ExecutionContext) -> List[dict[str, Any]]:
    applied = context.metadata.get("applied_persona_memories") if isinstance(context.metadata, dict) else None
    personas: list[dict[str, Any]] = []
    if isinstance(applied, dict):
        for entries in applied.values():
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if isinstance(entry, dict):
                    personas.append(entry)
    return personas


def _collect_persona_updates(context: ExecutionContext) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    candidates = context.metadata.get("new_persona_candidates") if isinstance(context.metadata, dict) else None
    if isinstance(candidates, list) and candidates:
        updates["new_candidates"] = candidates
    promotion = context.metadata.get("persona_promotion_result") if isinstance(context.metadata, dict) else None
    if promotion:
        updates["promotion_result"] = promotion
    return updates


def _extract_teacher_notes(context: ExecutionContext) -> List[str]:
    notes: list[str] = []
    steps = context.metadata.get("steps", {}) if isinstance(context.metadata, dict) else {}
    if isinstance(steps, dict):
        for meta in steps.values():
            if not isinstance(meta, dict):
                continue
            guidance = meta.get("guidance")
            if isinstance(guidance, list):
                for note in guidance:
                    if isinstance(note, str) and note.strip():
                        notes.append(note.strip())
    return notes


def _collect_reward_summary(result: Result) -> dict[str, Any]:
    rewards: list[float] = []
    for step in result.step_results:
        score = getattr(step.evaluation.reward, "score", None)
        if isinstance(score, (int, float)):
            rewards.append(float(score))
    return {
        "average": float(fmean(rewards)) if rewards else None,
        "count": len(rewards),
    }


def _infer_persona_from_step(step_result, step_metadata: dict[str, Any]) -> str:
    metadata = step_result.metadata or {}
    persona = metadata.get("persona_target") or metadata.get("persona") or metadata.get("actor")
    if isinstance(persona, str) and persona:
        return canonical_persona_name(persona)
    guidance_source = step_metadata.get("guidance_source")
    if isinstance(guidance_source, str) and guidance_source:
        return canonical_persona_name(guidance_source)
    return canonical_persona_name("student")


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
