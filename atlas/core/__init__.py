"""Atlas SDK public entry point."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import sys
import json
from statistics import fmean
from typing import Any
from typing import Dict
from typing import List
from typing import Protocol
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
from atlas.evaluation.evaluator import Evaluator
from atlas.personas.student import Student
from atlas.personas.teacher import Teacher
from atlas.runtime.storage.database import Database
from atlas.runtime.telemetry import ConsoleTelemetryStreamer
from atlas.runtime.telemetry.langchain_callback import configure_langchain_callbacks
from atlas.runtime.learning_history import aggregate_learning_history
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
    session_meta = execution_context.metadata.setdefault("session_metadata", {})
    learning_key = _build_learning_key(task, config, session_meta)
    session_meta["learning_key"] = learning_key
    execution_context.metadata["learning_key"] = learning_key
    database = Database(config.storage) if config.storage else None
    session_id: int | None = None
    try:
        if database:
            await database.connect()
            history_records = await database.fetch_learning_history(learning_key)
            learning_history = aggregate_learning_history(history_records)
            metadata = execution_context.metadata.get("session_metadata")
            session_id = await database.create_session(task, metadata=metadata)
            if publisher is not None and session_id is not None:
                publisher.publish_control_event(
                    "session-started",
                    {"session_id": session_id, "task": task},
                )
        else:
            learning_history = {}

        execution_context.metadata["learning_history"] = learning_history

        capability_probe_client = CapabilityProbeClient(adaptive_teaching_cfg.probe)

        orchestrator = Orchestrator(
            teacher=teacher,
            student=student,
            evaluator=evaluator,
            orchestration_config=config.orchestration,
            rim_config=config.rim,
            adaptive_config=adaptive_teaching_cfg,
            triage_adapter=triage_adapter,
            capability_probe=capability_probe_client,
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


def _build_learning_key(task: str, config: AtlasConfig, session_meta: Dict[str, Any]) -> str:
    existing_key = session_meta.get("learning_key")
    if isinstance(existing_key, str) and existing_key.strip():
        return existing_key.strip()
    override_key = session_meta.get("learning_key_override")
    if isinstance(override_key, str) and override_key.strip():
        return override_key.strip()
    agent_name = getattr(config.agent, "name", "agent")
    tenant_id = session_meta.get("tenant_id") or session_meta.get("tenant") or "default"
    raw_tags = session_meta.get("tags") or []
    if isinstance(raw_tags, str):
        tags = [raw_tags.strip()] if raw_tags.strip() else []
    elif isinstance(raw_tags, (list, tuple, set)):
        tags = [str(tag).strip() for tag in raw_tags if str(tag).strip()]
    else:
        tags = []
    payload = {
        "agent": agent_name,
        "tenant": str(tenant_id),
        "tags": sorted(tags),
        "task_prefix": task.strip()[:64],
    }
    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


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
    history_snapshot = context.metadata.get("learning_history") if isinstance(context.metadata, dict) else None
    if isinstance(history_snapshot, dict):
        payload["learning_history"] = history_snapshot
    learning_key = context.metadata.get("learning_key") if isinstance(context.metadata, dict) else None
    if learning_key:
        payload["learning_key"] = learning_key
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
    probe_payload = adaptive_meta.get("probe")
    if isinstance(probe_payload, dict):
        summary["probe"] = probe_payload
    return summary


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
