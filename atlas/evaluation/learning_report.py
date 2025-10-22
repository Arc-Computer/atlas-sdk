"""Utilities for building hint-less learning evaluation reports."""

from __future__ import annotations

import asyncio
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from statistics import fmean
from typing import Any, Iterable, Sequence

from atlas.runtime.storage.database import Database


@dataclass(slots=True)
class WindowSpec:
    label: str
    size: int


@dataclass(slots=True)
class SessionSnapshot:
    session_id: int
    created_at: str | None
    status: str | None
    review_status: str | None
    execution_mode: str | None
    reward_score: float | None
    reward_uncertainty: float | None
    reward_audit_count: int
    student_learning: str | None
    teacher_learning: str | None
    trajectory_events: int
    student_model_id: str | None = None
    teacher_model_id: str | None = None


@dataclass(slots=True)
class DiscoveryRunRef:
    run_id: int
    task: str | None
    source: str
    created_at: str | None


@dataclass(slots=True)
class RewardSnapshot:
    recent_mean: float | None
    recent_count: int
    baseline_mean: float | None
    baseline_count: int
    delta: float | None
    latest_score: float | None
    recent_window: WindowSpec | None = None
    baseline_window: WindowSpec | None = None


@dataclass(slots=True)
class LearningModelBreakdown:
    role: str
    model_id: str
    session_count: int
    reward_count: int
    reward_mean: float | None
    latest_score: float | None
    last_seen_at: str | None = None


@dataclass(slots=True)
class LearningSummary:
    learning_key: str
    session_count: int
    reward: RewardSnapshot
    recent_window: WindowSpec | None = None
    baseline_window: WindowSpec | None = None
    model_breakdown: list[LearningModelBreakdown] = field(default_factory=list)
    adaptive_modes: dict[str, int] = field(default_factory=dict)
    review_statuses: dict[str, int] = field(default_factory=dict)
    discovery_runs: list[DiscoveryRunRef] = field(default_factory=list)
    sessions: list[SessionSnapshot] = field(default_factory=list)


async def generate_learning_summary(
    database: Database,
    learning_key: str,
    *,
    recent_window: int | WindowSpec = 5,
    baseline_window: int | WindowSpec = 50,
    discovery_limit: int = 5,
    trajectory_limit: int = 200,
    summary_only: bool = False,
    session_limit: int | None = None,
    project_root: str | None = None,
    task_filter: str | None = None,
    tags: Sequence[str] | None = None,
) -> LearningSummary:
    return await _generate_learning_summary(
        database,
        learning_key,
        recent_window=recent_window,
        baseline_window=baseline_window,
        discovery_limit=discovery_limit,
        trajectory_limit=trajectory_limit,
        summary_only=summary_only,
        session_limit=session_limit,
        project_root=project_root,
        task_filter=task_filter,
        tags=tags,
    )


async def _generate_learning_summary(
    database: Database,
    learning_key: str,
    *,
    recent_window: int | WindowSpec = 5,
    baseline_window: int | WindowSpec = 50,
    discovery_limit: int = 5,
    trajectory_limit: int = 200,
    summary_only: bool = False,
    session_limit: int | None = None,
    project_root: str | None = None,
    task_filter: str | None = None,
    tags: Sequence[str] | None = None,
) -> LearningSummary:
    recent_spec = _coerce_window_spec(recent_window, default_label="recent")
    baseline_spec = _coerce_window_spec(baseline_window, default_label="baseline")
    rows = await database.fetch_learning_sessions(
        learning_key=learning_key,
        project_root=project_root,
        task=task_filter,
        tags=tags,
        limit=session_limit,
        order="asc",
    )
    sessions: list[SessionSnapshot] = []
    adaptive_counts: dict[str, int] = {}
    review_counts: dict[str, int] = {}
    reward_scores: list[float] = []
    tasks_seen: set[str] = set()
    model_accumulators: dict[tuple[str, str], dict[str, Any]] = {}
    session_ids = [row["id"] for row in rows if isinstance(row.get("id"), int)]
    trajectory_counts: dict[int, int] = {}
    if summary_only and session_ids:
        trajectory_counts = await database.fetch_trajectory_event_counts(session_ids)

    for row in rows:
        metadata = _coerce_dict(row.get("metadata"))
        reward_stats = _coerce_dict(row.get("reward_stats"))
        session_reward = _coerce_dict(row.get("reward"))
        reward_audit = _coerce_list(row.get("reward_audit"))
        execution_mode = _extract_execution_mode(metadata)
        if isinstance(execution_mode, str) and execution_mode:
            adaptive_counts[execution_mode] = adaptive_counts.get(execution_mode, 0) + 1
        review_status = row.get("review_status")
        if isinstance(review_status, str) and review_status:
            review_counts[review_status] = review_counts.get(review_status, 0) + 1
        reward_score = _extract_score(reward_stats, session_reward)
        reward_uncertainty = _extract_uncertainty(reward_stats, session_reward)
        if reward_score is not None:
            reward_scores.append(reward_score)
        created_at_raw = row.get("created_at")
        created_at = _format_timestamp(created_at_raw)
        if summary_only:
            trajectory_events = trajectory_counts.get(row["id"], 0)
        else:
            events = await database.fetch_trajectory_events(
                row["id"],
                limit=trajectory_limit,
            )
            trajectory_events = len(events)
        model_ids = _extract_model_ids(metadata)
        for role, model_id in model_ids.items():
            key = (role, model_id)
            accumulator = model_accumulators.setdefault(
                key,
                {"session_count": 0, "reward_count": 0, "reward_sum": 0.0, "latest_score": None, "last_seen_at": None},
            )
            accumulator["session_count"] += 1
            if reward_score is not None:
                accumulator["reward_count"] += 1
                accumulator["reward_sum"] += reward_score
                accumulator["latest_score"] = reward_score
            accumulator["last_seen_at"] = created_at
        snapshot = SessionSnapshot(
            session_id=row["id"],
            created_at=created_at,
            status=row.get("status"),
            review_status=review_status,
            execution_mode=execution_mode if isinstance(execution_mode, str) else None,
            reward_score=reward_score,
            reward_uncertainty=reward_uncertainty,
            reward_audit_count=len(reward_audit),
            student_learning=_trim_optional_str(row.get("student_learning")),
            teacher_learning=_trim_optional_str(row.get("teacher_learning")),
            trajectory_events=trajectory_events,
            student_model_id=model_ids.get("student"),
            teacher_model_id=model_ids.get("teacher"),
        )
        sessions.append(snapshot)
        task_value = row.get("task")
        if isinstance(task_value, str) and task_value.strip():
            tasks_seen.add(task_value)

    recent_scores = reward_scores[-recent_spec.size :] if recent_spec.size > 0 else reward_scores[:]
    recent_mean = fmean(recent_scores) if recent_scores else None
    baseline = await database.fetch_reward_baseline(learning_key, window=max(baseline_spec.size, 1))
    baseline_mean = _coerce_float(baseline.get("score_mean"))
    baseline_count = int(baseline.get("sample_count") or 0)
    latest_score = reward_scores[-1] if reward_scores else None
    delta = None
    if recent_mean is not None and baseline_mean is not None:
        delta = recent_mean - baseline_mean

    reward_snapshot = RewardSnapshot(
        recent_mean=recent_mean,
        recent_count=len(recent_scores),
        baseline_mean=baseline_mean,
        baseline_count=baseline_count,
        delta=delta,
        latest_score=latest_score,
        recent_window=recent_spec,
        baseline_window=baseline_spec,
    )

    discovery_refs = await _collect_discovery_refs(
        database,
        tasks_seen,
        limit=discovery_limit,
    )

    model_breakdown: list[LearningModelBreakdown] = []
    for (role, model_id), accumulator in sorted(model_accumulators.items(), key=lambda item: (item[0][0], item[0][1])):
        reward_count = accumulator["reward_count"]
        reward_mean = accumulator["reward_sum"] / reward_count if reward_count else None
        model_breakdown.append(
            LearningModelBreakdown(
                role=role,
                model_id=model_id,
                session_count=accumulator["session_count"],
                reward_count=reward_count,
                reward_mean=reward_mean,
                latest_score=accumulator["latest_score"],
                last_seen_at=accumulator["last_seen_at"],
            )
        )

    return LearningSummary(
        learning_key=learning_key,
        session_count=len(sessions),
        reward=reward_snapshot,
        recent_window=recent_spec,
        baseline_window=baseline_spec,
        model_breakdown=model_breakdown,
        adaptive_modes=dict(sorted(adaptive_counts.items())),
        review_statuses=dict(sorted(review_counts.items())),
        discovery_runs=discovery_refs,
        sessions=sessions,
    )


async def collect_learning_summaries(
    database: Database,
    learning_keys: Sequence[str],
    *,
    recent_window: int | WindowSpec = 5,
    baseline_window: int | WindowSpec = 50,
    discovery_limit: int = 5,
    trajectory_limit: int = 200,
    summary_only: bool = False,
    session_limit: int | None = None,
    project_root: str | None = None,
    task_filter: str | None = None,
    tags: Sequence[str] | None = None,
    max_concurrency: int = 4,
) -> list[LearningSummary]:
    if not learning_keys:
        return []

    semaphore: asyncio.Semaphore | None = None
    if max_concurrency and max_concurrency > 0:
        semaphore = asyncio.Semaphore(max_concurrency)

    async def _summarise(key: str) -> LearningSummary:
        return await _generate_learning_summary(
            database,
            key,
            recent_window=recent_window,
            baseline_window=baseline_window,
            discovery_limit=discovery_limit,
            trajectory_limit=trajectory_limit,
            summary_only=summary_only,
            session_limit=session_limit,
            project_root=project_root,
            task_filter=task_filter,
            tags=tags,
        )

    async def _task(key: str) -> LearningSummary:
        if semaphore is None:
            return await _summarise(key)
        async with semaphore:
            return await _summarise(key)

    tasks = [_task(key) for key in learning_keys]
    results = await asyncio.gather(*tasks)
    return list(results)


def summary_to_markdown(summary: LearningSummary) -> str:
    lines: list[str] = []
    lines.append(f"# Learning Evaluation — {summary.learning_key}")
    lines.append("")
    lines.append(f"- Sessions analysed: {summary.session_count}")
    if summary.reward.recent_window and summary.reward.recent_window.size:
        lines.append(f"- Recent window: last {summary.reward.recent_window.size} sessions")
    lines.append(
        "- Recent reward mean: "
        + (_format_float(summary.reward.recent_mean) if summary.reward.recent_mean is not None else "n/a")
    )
    if summary.reward.baseline_window and summary.reward.baseline_window.size:
        lines.append(f"- Baseline window: last {summary.reward.baseline_window.size} sessions")
    lines.append(
        "- Baseline reward mean: "
        + (_format_float(summary.reward.baseline_mean) if summary.reward.baseline_mean is not None else "n/a")
        + f" (n={summary.reward.baseline_count})"
    )
    if summary.reward.delta is not None:
        direction = "improved" if summary.reward.delta >= 0 else "regressed"
        lines.append(f"- Reward delta vs baseline: {_format_float(summary.reward.delta)} ({direction})")
    if summary.reward.latest_score is not None:
        lines.append(f"- Latest reward score: {_format_float(summary.reward.latest_score)}")
    if summary.adaptive_modes:
        modes = ", ".join(f"{mode}: {count}" for mode, count in summary.adaptive_modes.items())
        lines.append(f"- Adaptive modes observed: {modes}")
    if summary.review_statuses:
        statuses = ", ".join(f"{status}: {count}" for status, count in summary.review_statuses.items())
        lines.append(f"- Review statuses: {statuses}")
    if summary.discovery_runs:
        lines.append("- Discovery telemetry references:")
        for ref in summary.discovery_runs:
            timestamp = ref.created_at or "unknown"
            lines.append(f"  - #{ref.run_id} [{ref.source}] task={ref.task!r} at {timestamp}")
    if summary.model_breakdown:
        lines.append("")
        lines.append("## Model Performance")
        for entry in summary.model_breakdown:
            reward_mean = _format_float(entry.reward_mean)
            latest = _format_float(entry.latest_score)
            last_seen = entry.last_seen_at or "n/a"
            lines.append(
                f"- {entry.role.title()} model `{entry.model_id}` — sessions={entry.session_count}, "
                f"reward_mean={reward_mean}, latest={latest}, last_seen={last_seen}"
            )
    lines.append("")
    lines.append("## Latest Sessions")
    if not summary.sessions:
        lines.append("No sessions found for this learning key.")
        return "\n".join(lines)
    for snapshot in summary.sessions[-10:]:
        model_notes = []
        if snapshot.student_model_id:
            model_notes.append(f"student={snapshot.student_model_id}")
        if snapshot.teacher_model_id:
            model_notes.append(f"teacher={snapshot.teacher_model_id}")
        model_suffix = f", models={' / '.join(model_notes)}" if model_notes else ""
        lines.append(
            f"- Session {snapshot.session_id} ({snapshot.created_at or 'unknown'}): "
            f"mode={snapshot.execution_mode or 'n/a'}, "
            f"score={_format_float(snapshot.reward_score)}, "
            f"uncertainty={_format_float(snapshot.reward_uncertainty)}, "
            f"review={snapshot.review_status or 'n/a'}, "
            f"trajectory_events={snapshot.trajectory_events}"
            f"{model_suffix}"
        )
        learning_details: list[str] = []
        if snapshot.student_learning:
            learning_details.append(f"student learning: {snapshot.student_learning}")
        if snapshot.teacher_learning:
            learning_details.append(f"teacher learning: {snapshot.teacher_learning}")
        if learning_details:
            for detail in learning_details:
                lines.append(f"  - {detail}")
    return "\n".join(lines)


def summary_to_dict(summary: LearningSummary) -> dict[str, Any]:
    return asdict(summary)


async def _collect_discovery_refs(
    database: Database,
    tasks: Iterable[str],
    *,
    limit: int,
) -> list[DiscoveryRunRef]:
    refs: list[DiscoveryRunRef] = []
    seen_ids: set[int] = set()
    for task in tasks:
        runs = await database.fetch_discovery_runs(
            task=task,
            source=["discovery", "runtime"],
            limit=limit,
        )
        for entry in runs:
            run_id = entry.get("id")
            if not isinstance(run_id, int) or run_id in seen_ids:
                continue
            seen_ids.add(run_id)
            refs.append(
                DiscoveryRunRef(
                    run_id=run_id,
                    task=entry.get("task"),
                    source=str(entry.get("source") or "unknown"),
                    created_at=_format_timestamp(entry.get("created_at")),
                )
            )
    refs.sort(key=lambda ref: ref.created_at or "", reverse=True)
    return refs[:limit]


def _coerce_window_spec(value: int | WindowSpec, *, default_label: str) -> WindowSpec:
    if isinstance(value, WindowSpec):
        size = max(int(value.size), 0)
        label = value.label or default_label
        return WindowSpec(label=label, size=size)
    try:
        size = max(int(value), 0)
    except (TypeError, ValueError):
        size = 0
    return WindowSpec(label=default_label, size=size)


def _extract_execution_mode(metadata: dict[str, Any]) -> str | None:
    execution_mode = metadata.get("execution_mode")
    if isinstance(execution_mode, str) and execution_mode.strip():
        return execution_mode.strip()
    summary = metadata.get("adaptive_summary")
    if isinstance(summary, dict):
        summary_mode = summary.get("adaptive_mode")
        if isinstance(summary_mode, str) and summary_mode.strip():
            return summary_mode.strip()
    return None


def _extract_model_ids(metadata: dict[str, Any]) -> dict[str, str]:
    models: dict[str, str] = {}
    adapter_session = metadata.get("adapter_session")
    if isinstance(adapter_session, dict):
        student_model = adapter_session.get("student_model_id") or adapter_session.get("student_model")
        teacher_model = adapter_session.get("teacher_model_id") or adapter_session.get("teacher_model")
        if isinstance(student_model, str) and student_model.strip():
            models["student"] = student_model.strip()
        if isinstance(teacher_model, str) and teacher_model.strip():
            models["teacher"] = teacher_model.strip()
    # Fallbacks for legacy metadata placements.
    student_direct = metadata.get("student_model")
    if "student" not in models and isinstance(student_direct, str) and student_direct.strip():
        models["student"] = student_direct.strip()
    teacher_direct = metadata.get("teacher_model")
    if "teacher" not in models and isinstance(teacher_direct, str) and teacher_direct.strip():
        models["teacher"] = teacher_direct.strip()
    return models


def _coerce_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return dict(payload)
    if isinstance(payload, str):
        return _parse_json_dict(payload)
    return {}


def _coerce_list(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return list(payload)
    if isinstance(payload, str):
        parsed = _parse_json(payload)
        return parsed if isinstance(parsed, list) else []
    return []


def _parse_json(payload: str) -> Any:
    import json

    try:
        return json.loads(payload)
    except (TypeError, ValueError):
        return None


def _parse_json_dict(payload: str) -> dict[str, Any]:
    parsed = _parse_json(payload)
    return dict(parsed) if isinstance(parsed, dict) else {}


def _extract_score(reward_stats: dict[str, Any], session_reward: dict[str, Any]) -> float | None:
    for source in (reward_stats, session_reward):
        value = source.get("score") if isinstance(source, dict) else None
        if value is not None:
            return _coerce_float(value)
    return None


def _extract_uncertainty(reward_stats: dict[str, Any], session_reward: dict[str, Any]) -> float | None:
    candidates = [
        reward_stats.get("uncertainty_mean"),
        reward_stats.get("uncertainty"),
        session_reward.get("uncertainty") if isinstance(session_reward, dict) else None,
    ]
    for value in candidates:
        result = _coerce_float(value)
        if result is not None:
            return result
    return None


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
        if math.isnan(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def _trim_optional_str(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _format_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"
