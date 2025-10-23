"""Atlas SDK public entry point."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import sys
import json
from datetime import datetime, timezone
from statistics import fmean
from typing import Any
from typing import Dict
from typing import List
from typing import Protocol
from importlib import import_module

from atlas.connectors.factory import create_from_atlas_config
from atlas.config.loader import load_config
from atlas.config.models import AdaptiveTeachingConfig, AtlasConfig, LearningConfig, RewardObjectiveConfig, RuntimeSafetyConfig
from atlas.prompts import (
    RewrittenStudentPrompts as RewrittenStudentPrompts,
    RewrittenTeacherPrompts as RewrittenTeacherPrompts,
    build_student_prompts,
    build_teacher_prompts,
)
from atlas.runtime.orchestration.execution_context import ExecutionContext
from atlas.runtime.adaptive import CapabilityProbeClient
from atlas.runtime.orchestration.orchestrator import Orchestrator
from atlas.evaluation.evaluator import Evaluator
from atlas.personas.student import Student
from atlas.personas.teacher import Teacher
from atlas.learning import LearningSynthesizer
from atlas.runtime.storage.database import Database
from atlas.runtime.learning.drift import RewardDriftDetector
from atlas.runtime.telemetry import ConsoleTelemetryStreamer
from atlas.runtime.telemetry.langchain_callback import configure_langchain_callbacks
from atlas.runtime.learning_history import DEFAULT_HISTORY_LIMIT, aggregate_learning_history
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
    learning_cfg = getattr(config, "learning", LearningConfig())
    apply_learning_prompts = getattr(learning_cfg, "apply_to_prompts", True)
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
    execution_context.metadata["learning_apply_to_prompts"] = apply_learning_prompts
    student = _build_student(
        adapter,
        config,
        base_student_prompts,
        apply_learning_prompts=apply_learning_prompts,
    )
    teacher = Teacher(
        config.teacher,
        base_teacher_prompts,
        adapter_config.tools,
        apply_learning_prompts=apply_learning_prompts,
    )
    evaluator = _build_evaluator_instance(config, getattr(adaptive_teaching_cfg, "reward", None))
    learning_synthesizer = _build_learning_synthesizer(config)
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
            learning_state = await database.fetch_learning_state(learning_key)
            history_records = await database.fetch_learning_history(learning_key)
            learning_history = aggregate_learning_history(
                history_records,
                limit=getattr(adaptive_teaching_cfg, "learning_history_limit", DEFAULT_HISTORY_LIMIT),
            )
            metadata = execution_context.metadata.get("session_metadata")
            session_id = await database.create_session(task, metadata=metadata)
            if publisher is not None and session_id is not None:
                publisher.publish_control_event(
                    "session-started",
                    {"session_id": session_id, "task": task},
                )
        else:
            learning_history = {}
            learning_state = {}

        execution_context.metadata["learning_history"] = learning_history
        execution_context.metadata["learning_state"] = learning_state or {}

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
        if (
            database
            and learning_synthesizer
            and learning_synthesizer.enabled
            and learning_cfg.update_enabled
            and execution_context.metadata.get("session_reward") is not None
        ):
            reward_payload = execution_context.metadata.get("session_reward")
            trajectory_payload = execution_context.metadata.get("session_trajectory")
            history_payload = execution_context.metadata.get("learning_history")
            current_learning_state = execution_context.metadata.get("learning_state") or {}
            synthesis = await learning_synthesizer.asynthesize(
                learning_key=learning_key,
                task=task,
                reward=reward_payload if isinstance(reward_payload, dict) else _safe_reward_to_dict(reward_payload),
                trajectory=trajectory_payload if isinstance(trajectory_payload, dict) else None,
                learning_state=current_learning_state if isinstance(current_learning_state, dict) else {},
                history=history_payload if isinstance(history_payload, dict) else None,
            )
            if synthesis is not None:
                session_note = synthesis.session_note if learning_cfg.session_note_enabled else None
                execution_context.set_session_learning(
                    student_learning=synthesis.student_learning,
                    teacher_learning=synthesis.teacher_learning,
                    learning_state=synthesis.learning_state,
                    session_note=session_note,
                )
        if database and session_id is not None:
            await _persist_results(
                database,
                session_id,
                execution_context,
                result,
                events,
                runtime_safety=config.runtime_safety,
                )
            if learning_cfg.enabled and learning_cfg.update_enabled:
                updated_state = execution_context.metadata.get("learning_state")
                if isinstance(updated_state, dict) and updated_state:
                    await database.upsert_learning_state(
                        learning_key,
                        updated_state.get("student_learning"),
                        updated_state.get("teacher_learning"),
                        updated_state.get("metadata"),
                    )
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
            await _persist_failure_metadata(database, session_id, execution_context)
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


def _build_student(
    adapter,
    config: AtlasConfig,
    student_prompts,
    *,
    apply_learning_prompts: bool = True,
) -> Student:
    adapter_config = config.agent
    return Student(
        adapter=adapter,
        adapter_config=adapter_config,
        student_config=config.student,
        student_prompts=student_prompts,
        apply_learning_prompts=apply_learning_prompts,
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


def _build_learning_synthesizer(config: AtlasConfig) -> LearningSynthesizer | None:
    learning_cfg = getattr(config, "learning", None)
    if learning_cfg is None or not learning_cfg.enabled:
        return None
    fallback_llm = getattr(config.rim, "large_model", None)
    return LearningSynthesizer(learning_cfg, fallback_llm=fallback_llm)


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


def _safe_reward_to_dict(payload: Any) -> Dict[str, Any]:
    if payload is None:
        return {}
    if hasattr(payload, "to_dict"):
        try:
            return payload.to_dict()
        except Exception:  # pragma: no cover - defensive guard
            return {}
    if isinstance(payload, dict):
        return payload
    return {}


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
    *,
    runtime_safety: RuntimeSafetyConfig | None = None,
) -> None:
    review_cfg = runtime_safety.review if runtime_safety is not None else None
    require_review_approval = True
    default_export_statuses: list[str] = ["approved"]
    include_all_default_statuses = False
    if review_cfg is not None:
        require_review_approval = bool(review_cfg.require_approval)
        configured_statuses = [
            str(status).strip().lower()
            for status in getattr(review_cfg, "default_export_statuses", []) or []
            if str(status).strip()
        ]
        if configured_statuses:
            default_export_statuses = configured_statuses
        else:
            include_all_default_statuses = True
    env_review_required = os.getenv("ATLAS_REVIEW_REQUIRE_APPROVAL")
    if env_review_required is not None:
        require_review_approval = env_review_required.strip().lower() not in {"0", "false", "off"}
    env_default_statuses = os.getenv("ATLAS_REVIEW_DEFAULT_EXPORT_STATUSES")
    if env_default_statuses:
        tokens = [token.strip().lower() for token in env_default_statuses.split(",") if token.strip()]
        if any(token in {"*", "all"} for token in tokens):
            include_all_default_statuses = True
            default_export_statuses = []
        elif tokens:
            include_all_default_statuses = False
            default_export_statuses = tokens
    session_meta = context.metadata.setdefault("session_metadata", {})
    review_meta = session_meta.setdefault("review", {})
    review_meta["require_approval"] = require_review_approval
    review_meta["default_export_statuses"] = list(default_export_statuses)
    review_meta["include_all_statuses"] = include_all_default_statuses

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
    reward_stats = context.metadata.get("session_reward_stats") if isinstance(context.metadata, dict) else None
    reward_audit = context.metadata.get("session_reward_audit") if isinstance(context.metadata, dict) else None
    stats_payload: dict[str, Any] | None = None
    audit_payload: list[dict[str, Any]] | None = None
    if isinstance(reward_stats, dict):
        stats_payload = dict(reward_stats)
        stats_payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        context.metadata["session_reward_stats"] = stats_payload
    if isinstance(reward_audit, list):
        audit_payload = [dict(entry) for entry in reward_audit]
        context.metadata["session_reward_audit"] = audit_payload
    if session_reward is not None or student_learning is not None or teacher_learning is not None:
        await database.log_session_reward(
            session_id,
            session_reward,
            student_learning,
            teacher_learning,
            stats_payload,
            audit_payload,
        )
    drift_result = None
    drift_cfg = runtime_safety.drift if runtime_safety is not None else None
    drift_enabled = True
    drift_window = 50
    drift_threshold = 3.0
    drift_min_baseline = 5
    if drift_cfg is not None:
        drift_enabled = bool(drift_cfg.enabled)
        drift_window = drift_cfg.window
        drift_threshold = drift_cfg.z_threshold
        drift_min_baseline = drift_cfg.min_baseline
    env_enabled = os.getenv("ATLAS_DRIFT_ENABLED")
    if env_enabled is not None:
        drift_enabled = env_enabled.strip().lower() not in {"0", "false", "off"}
    env_window = os.getenv("ATLAS_DRIFT_WINDOW")
    if env_window:
        try:
            drift_window = max(int(env_window), 1)
        except ValueError:
            logger.warning("Invalid ATLAS_DRIFT_WINDOW value '%s'; using %s", env_window, drift_window)
    env_threshold = os.getenv("ATLAS_DRIFT_Z_THRESHOLD")
    if env_threshold:
        try:
            drift_threshold = max(float(env_threshold), 0.0)
        except ValueError:
            logger.warning("Invalid ATLAS_DRIFT_Z_THRESHOLD value '%s'; using %s", env_threshold, drift_threshold)
    env_min_baseline = os.getenv("ATLAS_DRIFT_MIN_BASELINE")
    if env_min_baseline:
        try:
            drift_min_baseline = max(int(env_min_baseline), 0)
        except ValueError:
            logger.warning(
                "Invalid ATLAS_DRIFT_MIN_BASELINE value '%s'; using %s",
                env_min_baseline,
                drift_min_baseline,
            )

    if drift_enabled and stats_payload is not None:
        drift_detector = RewardDriftDetector(
            window=drift_window,
            z_threshold=drift_threshold,
            min_baseline=drift_min_baseline,
        )
        learning_key = session_meta.get("learning_key") or context.metadata.get("learning_key")
        drift_result = await drift_detector.assess(
            database,
            learning_key=learning_key,
            current_stats=stats_payload,
        )
        if drift_result is not None:
            drift_payload = drift_result.to_dict()
            session_meta["drift"] = drift_payload
            session_meta["drift_alert"] = drift_payload.get("drift_alert", False)
            if drift_result.alert:
                drift_payload.setdefault("message", f"Reward drift alert ({drift_result.reason or 'threshold'})")
        context.metadata["session_metadata"] = session_meta
    review_status_update: str | None = None
    review_notes_update: str | None = None
    if drift_result is not None and drift_result.alert:
        review_status_update = "pending"
        review_notes_update = session_meta.get("drift", {}).get("message") or (drift_result.reason or None)
    elif not require_review_approval:
        review_status_update = "approved"
        review_notes_update = "Auto-approved (review gating disabled)."
    if review_status_update is not None:
        await database.update_session_review_status(session_id, review_status_update, review_notes_update)
        session_meta["review_status"] = review_status_update
        if review_notes_update:
            session_meta["review_notes"] = review_notes_update
        context.metadata["session_metadata"] = session_meta
    if audit_payload is not None:
        session_meta = context.metadata.setdefault("session_metadata", {})
        session_meta["reward_audit"] = audit_payload
        context.metadata["session_metadata"] = session_meta
    await _update_session_metadata(database, session_id, context, result)
    await _persist_events(database, session_id, events)


async def _persist_events(database: Database, session_id: int, events: List) -> None:
    for event in events:
        await database.log_intermediate_step(session_id, event)


async def _persist_failure_metadata(database: Database, session_id: int, context: ExecutionContext) -> None:
    base_metadata = context.metadata.get("session_metadata") or {}
    if not isinstance(base_metadata, dict):
        base_metadata = {}
    insights = _collect_session_insights(context, None)
    if not insights and not base_metadata:
        return
    merged = {**base_metadata, **insights}
    context.metadata["session_metadata"] = merged
    await database.update_session_metadata(session_id, merged)


async def _update_session_metadata(
    database: Database,
    session_id: int,
    context: ExecutionContext,
    result: Result | None,
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


def _collect_session_insights(context: ExecutionContext, result: Result | None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    triage = context.metadata.get("triage", {}).get("dossier") if isinstance(context.metadata, dict) else None
    if triage:
        payload["triage_dossier"] = triage
    adaptive_summary = _collect_adaptive_summary(context)
    if adaptive_summary:
        context.metadata["adaptive_summary"] = adaptive_summary
        payload["adaptive_summary"] = adaptive_summary
    execution_mode = None
    if isinstance(context.metadata, dict):
        raw_mode = context.metadata.get("execution_mode")
        if isinstance(raw_mode, str) and raw_mode.strip():
            execution_mode = raw_mode.strip()
    if execution_mode is None and adaptive_summary:
        summary_mode = adaptive_summary.get("adaptive_mode")
        if isinstance(summary_mode, str) and summary_mode.strip():
            execution_mode = summary_mode.strip()
    if execution_mode:
        payload["execution_mode"] = execution_mode
    session_reward = context.metadata.get("session_reward") if isinstance(context.metadata, dict) else None
    if session_reward is not None:
        reward_payload = session_reward.to_dict() if hasattr(session_reward, "to_dict") else session_reward
        payload["session_reward"] = reward_payload
    reward_stats = context.metadata.get("session_reward_stats") if isinstance(context.metadata, dict) else None
    reward_audit = context.metadata.get("session_reward_audit") if isinstance(context.metadata, dict) else None
    if isinstance(reward_stats, dict):
        payload["reward_stats"] = dict(reward_stats)
    if isinstance(reward_audit, list):
        payload["reward_audit"] = [dict(entry) for entry in reward_audit if isinstance(entry, dict)]
    student_learning = context.metadata.get("session_student_learning") if isinstance(context.metadata, dict) else None
    if isinstance(student_learning, str) and student_learning.strip():
        payload["student_learning"] = student_learning
    teacher_learning = context.metadata.get("session_teacher_learning") if isinstance(context.metadata, dict) else None
    if isinstance(teacher_learning, str) and teacher_learning.strip():
        payload["teacher_learning"] = teacher_learning
    session_note = context.metadata.get("session_learning_note") if isinstance(context.metadata, dict) else None
    if isinstance(session_note, str) and session_note.strip():
        payload["session_learning_note"] = session_note.strip()
    learning_state_snapshot = context.metadata.get("learning_state") if isinstance(context.metadata, dict) else None
    if isinstance(learning_state_snapshot, dict) and learning_state_snapshot:
        payload["learning_state"] = learning_state_snapshot
    session_reward_payload = payload.get("session_reward")
    reward_summary: dict[str, Any] | None = None
    if session_reward_payload:
        raw_score = None
        if isinstance(session_reward_payload, dict):
            raw_score = session_reward_payload.get("score")
        reward_summary = {"score": raw_score}
    elif result is not None:
        reward_summary = _collect_reward_summary(result)
    if reward_summary is not None:
        if isinstance(reward_stats, dict):
            reward_summary.setdefault("score", reward_stats.get("score"))
            if reward_stats.get("score_stddev") is not None:
                reward_summary["score_stddev"] = reward_stats.get("score_stddev")
            if reward_stats.get("uncertainty_mean") is not None:
                reward_summary["uncertainty_mean"] = reward_stats.get("uncertainty_mean")
            if reward_stats.get("uncertainty_stddev") is not None:
                reward_summary["uncertainty_stddev"] = reward_stats.get("uncertainty_stddev")
        payload["reward_summary"] = reward_summary
    history_snapshot = context.metadata.get("learning_history") if isinstance(context.metadata, dict) else None
    if isinstance(history_snapshot, dict):
        payload["learning_history"] = history_snapshot
    learning_key = context.metadata.get("learning_key") if isinstance(context.metadata, dict) else None
    if learning_key:
        payload["learning_key"] = learning_key
    teacher_notes = _extract_teacher_notes(context)
    if teacher_notes:
        payload["teacher_notes"] = teacher_notes
    adapter_session = context.metadata.get("adapter_session") if isinstance(context.metadata, dict) else None
    if isinstance(adapter_session, dict) and adapter_session:
        payload["adapter_session"] = adapter_session
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
