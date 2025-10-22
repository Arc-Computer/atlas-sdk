"""Utilities for building hint-less learning evaluation reports."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from statistics import fmean
from typing import Any, Iterable, Sequence

from atlas.runtime.storage.database import Database


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


@dataclass(slots=True)
class LearningSummary:
    learning_key: str
    session_count: int
    reward: RewardSnapshot
    adaptive_modes: dict[str, int] = field(default_factory=dict)
    review_statuses: dict[str, int] = field(default_factory=dict)
    discovery_runs: list[DiscoveryRunRef] = field(default_factory=list)
    sessions: list[SessionSnapshot] = field(default_factory=list)


async def generate_learning_summary(
    database: Database,
    learning_key: str,
    *,
    recent_window: int = 5,
    baseline_window: int = 50,
    discovery_limit: int = 5,
    trajectory_limit: int = 200,
) -> LearningSummary:
    rows = await database.fetch_sessions_for_learning_key(learning_key)
    sessions: list[SessionSnapshot] = []
    adaptive_counts: dict[str, int] = {}
    review_counts: dict[str, int] = {}
    reward_scores: list[float] = []
    reward_uncertainties: list[float] = []
    tasks_seen: set[str] = set()

    for row in rows:
        metadata = _coerce_dict(row.get("metadata"))
        reward_stats = _coerce_dict(row.get("reward_stats"))
        session_reward = _coerce_dict(row.get("reward"))
        reward_audit = _coerce_list(row.get("reward_audit"))
        execution_mode = metadata.get("execution_mode")
        if not execution_mode:
            summary = metadata.get("adaptive_summary")
            if isinstance(summary, dict):
                execution_mode = summary.get("adaptive_mode")
        if isinstance(execution_mode, str) and execution_mode:
            adaptive_counts[execution_mode] = adaptive_counts.get(execution_mode, 0) + 1
        review_status = row.get("review_status")
        if isinstance(review_status, str) and review_status:
            review_counts[review_status] = review_counts.get(review_status, 0) + 1
        reward_score = _extract_score(reward_stats, session_reward)
        reward_uncertainty = _extract_uncertainty(reward_stats, session_reward)
        if reward_score is not None:
            reward_scores.append(reward_score)
        if reward_uncertainty is not None:
            reward_uncertainties.append(reward_uncertainty)
        created_at = _format_timestamp(row.get("created_at"))
        trajectory_events = await database.fetch_trajectory_events(
            row["id"],
            limit=trajectory_limit,
        )
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
            trajectory_events=len(trajectory_events),
        )
        sessions.append(snapshot)
        task_value = row.get("task")
        if isinstance(task_value, str) and task_value.strip():
            tasks_seen.add(task_value)

    recent_scores = reward_scores[-recent_window:] if recent_window > 0 else reward_scores[:]
    recent_mean = fmean(recent_scores) if recent_scores else None
    baseline = await database.fetch_reward_baseline(learning_key, window=baseline_window)
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
    )

    discovery_refs = await _collect_discovery_refs(
        database,
        tasks_seen,
        limit=discovery_limit,
    )

    return LearningSummary(
        learning_key=learning_key,
        session_count=len(sessions),
        reward=reward_snapshot,
        adaptive_modes=dict(sorted(adaptive_counts.items())),
        review_statuses=dict(sorted(review_counts.items())),
        discovery_runs=discovery_refs,
        sessions=sessions,
    )


async def collect_learning_summaries(
    database: Database,
    learning_keys: Sequence[str],
    *,
    recent_window: int = 5,
    baseline_window: int = 50,
    discovery_limit: int = 5,
    trajectory_limit: int = 200,
) -> list[LearningSummary]:
    summaries: list[LearningSummary] = []
    for key in learning_keys:
        summary = await generate_learning_summary(
            database,
            key,
            recent_window=recent_window,
            baseline_window=baseline_window,
            discovery_limit=discovery_limit,
            trajectory_limit=trajectory_limit,
        )
        summaries.append(summary)
    return summaries


def summary_to_markdown(summary: LearningSummary) -> str:
    lines: list[str] = []
    lines.append(f"# Learning Evaluation — {summary.learning_key}")
    lines.append("")
    lines.append(f"- Sessions analysed: {summary.session_count}")
    lines.append(
        "- Recent reward mean: "
        + (_format_float(summary.reward.recent_mean) if summary.reward.recent_mean is not None else "n/a")
    )
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
    lines.append("")
    lines.append("## Latest Sessions")
    if not summary.sessions:
        lines.append("No sessions found for this learning key.")
        return "\n".join(lines)
    for snapshot in summary.sessions[-10:]:
        lines.append(
            f"- Session {snapshot.session_id} ({snapshot.created_at or 'unknown'}): "
            f"mode={snapshot.execution_mode or 'n/a'}, "
            f"score={_format_float(snapshot.reward_score)}, "
            f"uncertainty={_format_float(snapshot.reward_uncertainty)}, "
            f"review={snapshot.review_status or 'n/a'}, "
            f"trajectory_events={snapshot.trajectory_events}"
        )
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
