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
    token_usage: dict[str, Any] | None = None
    learning_usage: dict[str, Any] | None = None


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
class PlaybookMetricsSummary:
    total_candidates: int
    passed: int
    failed: int
    pass_rate: float | None
    gate_failures: dict[str, int]
    average_weighted_score: float | None
    weights: dict[str, float] | None = None


@dataclass(slots=True)
class PlaybookLifecycleSummary:
    reinforcement_active: int
    reinforcement_deprecated: int
    differentiation_active: int
    differentiation_deprecated: int
    rejected: int


@dataclass(slots=True)
class PlaybookImpactEntry:
    entry_id: str
    audience: str | None
    cue_pattern: str | None
    runtime_handle: str | None
    total_cue_hits: int
    adoption_events: int
    successful_adoptions: int
    adoption_rate: float | None
    average_reward_with: float | None
    average_reward_without: float | None
    reward_delta: float | None
    average_tokens_with: float | None
    average_tokens_without: float | None
    token_delta: float | None
    unique_incidents: int
    transfer_success: bool
    transfer_level: str
    failure_avoidance: dict[str, Any] | None = None
    impact_score: float | None = None


@dataclass(slots=True)
class UsageMetrics:
    cue_triggers: int
    cue_trigger_sessions: int
    unique_cue_steps: int
    adoption_events: int
    successful_adoptions: int
    adoption_rate: float | None
    cue_trigger_rate: float | None


@dataclass(slots=True)
class EfficiencySnapshot:
    sessions_with_cues: int
    sessions_without_cues: int
    average_reward_with_cues: float | None
    average_reward_without_cues: float | None
    average_tokens_with_cues: float | None
    average_tokens_without_cues: float | None
    reward_delta: float | None
    token_delta: float | None


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
    playbook_metrics: PlaybookMetricsSummary | None = None
    playbook_lifecycle_summary: PlaybookLifecycleSummary | None = None
    playbook_impact: list[PlaybookImpactEntry] = field(default_factory=list)
    usage_metrics: UsageMetrics | None = None
    efficiency: EfficiencySnapshot | None = None


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
    latest_playbook_meta: dict[str, Any] | None = None
    rejected_candidates_total = 0
    cue_stats = {"cue_triggers": 0, "cue_sessions": 0, "unique_cue_steps": 0, "adoptions": 0, "success": 0}
    reward_with_cues: list[float] = []
    reward_without_cues: list[float] = []
    tokens_with_cues: list[float] = []
    tokens_without_cues: list[float] = []

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
        token_usage = _coerce_dict(metadata.get("token_usage"))
        learning_usage_meta = _coerce_dict(metadata.get("learning_usage"))
        learning_state_meta = _coerce_dict(metadata.get("learning_state"))
        playbook_meta = _coerce_dict(learning_state_meta.get("metadata")) if learning_state_meta else {}
        if playbook_meta:
            latest_playbook_meta = playbook_meta
            failure_meta = _coerce_dict(playbook_meta.get("last_failure"))
            rejected_list = _coerce_list(failure_meta.get("rejected_candidates")) if failure_meta else []
            rejected_candidates_total = len(rejected_list)
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
        session_has_cues = False
        if learning_usage_meta:
            session_block = _coerce_dict(learning_usage_meta.get("session"))
            cue_hits = int(session_block.get("cue_hits") or 0)
            adoption_events = int(session_block.get("action_adoptions") or 0)
            unique_cue_steps = len(session_block.get("unique_cue_steps") or [])
            success_count = 0
            roles_usage = learning_usage_meta.get("roles")
            if isinstance(roles_usage, dict):
                for role_usage in roles_usage.values():
                    if not isinstance(role_usage, dict):
                        continue
                    for entry in role_usage.values():
                        if isinstance(entry, dict):
                            success_count += int(entry.get("successful_adoptions") or 0)
            cue_stats["cue_triggers"] += cue_hits
            cue_stats["unique_cue_steps"] += unique_cue_steps
            cue_stats["adoptions"] += adoption_events
            cue_stats["success"] += success_count
            if cue_hits > 0:
                cue_stats["cue_sessions"] += 1
                session_has_cues = True
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
            token_usage=token_usage if token_usage else None,
            learning_usage=learning_usage_meta if learning_usage_meta else None,
        )
        sessions.append(snapshot)
        task_value = row.get("task")
        if isinstance(task_value, str) and task_value.strip():
            tasks_seen.add(task_value)
        token_total = _coerce_float(token_usage.get("total_tokens")) if token_usage else None
        if reward_score is not None:
            if session_has_cues:
                reward_with_cues.append(reward_score)
            else:
                reward_without_cues.append(reward_score)
        if token_total is not None:
            if session_has_cues:
                tokens_with_cues.append(token_total)
            else:
                tokens_without_cues.append(token_total)

    playbook_metrics = _build_playbook_metrics(latest_playbook_meta)
    playbook_lifecycle_summary = _build_playbook_lifecycle_summary(latest_playbook_meta, rejected_candidates_total)
    playbook_impact = _build_playbook_impact(latest_playbook_meta)
    usage_metrics = _build_usage_metrics(cue_stats, len(sessions))
    efficiency_snapshot = _build_efficiency_snapshot(
        reward_with_cues,
        reward_without_cues,
        tokens_with_cues,
        tokens_without_cues,
    )

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
        playbook_metrics=playbook_metrics,
        playbook_lifecycle_summary=playbook_lifecycle_summary,
        playbook_impact=playbook_impact,
        usage_metrics=usage_metrics,
        efficiency=efficiency_snapshot,
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
    if summary.playbook_metrics:
        metrics = summary.playbook_metrics
        lines.append("")
        lines.append("## Playbook Entry Quality")
        lines.append(
            f"- Candidates evaluated: {metrics.total_candidates} (pass rate: {_format_float(metrics.pass_rate)}; passed={metrics.passed}, failed={metrics.failed})"
        )
        if metrics.average_weighted_score is not None:
            lines.append(f"- Average weighted score: {_format_float(metrics.average_weighted_score)}")
        if metrics.gate_failures:
            failures = ", ".join(f"{name}: {count}" for name, count in metrics.gate_failures.items())
            lines.append(f"- Gate failures: {failures}")
    if summary.playbook_lifecycle_summary:
        lifecycle = summary.playbook_lifecycle_summary
        lines.append("")
        lines.append("## Playbook Entry Lifecycle")
        lines.append(
            "- Reinforcement — active: {0}, deprecated: {1}".format(
                lifecycle.reinforcement_active,
                lifecycle.reinforcement_deprecated,
            )
        )
        lines.append(
            "- Differentiation — active: {0}, deprecated: {1}".format(
                lifecycle.differentiation_active,
                lifecycle.differentiation_deprecated,
            )
        )
        lines.append(f"- Rejected candidates (latest run): {lifecycle.rejected}")
    if summary.playbook_impact:
        lines.append("")
        lines.append("## Playbook Entry Impact")
        top_entries = summary.playbook_impact[:10]
        for entry in top_entries:
            adoption = _format_percent(entry.adoption_rate)
            reward_delta = _format_float(entry.reward_delta)
            token_delta = _format_float(entry.token_delta)
            impact_score = _format_float(entry.impact_score)
            transfer = "yes" if entry.transfer_success else "no"
            transfer_display = f"{transfer} ({entry.transfer_level})"
            lines.append(
                f"- `{entry.entry_id}` ({entry.audience or 'student'}) — hits {entry.total_cue_hits}, adoptions {entry.adoption_events}, adoption rate {adoption}, reward Δ {reward_delta}, tokens Δ {token_delta}, transfer {transfer_display}, impact score {impact_score}"
            )
            failure_stats = entry.failure_avoidance or {}
            if failure_stats:
                retry_avg = _format_float(failure_stats.get("retry_avg"))
                retry_samples = failure_stats.get("retry_samples", 0)
                failure_events = failure_stats.get("failure_events", 0)
                lines.append(
                    f"  - Failure avoidance: retry avg {retry_avg} across {retry_samples} sessions; failure events {failure_events} (failed adoptions {failure_stats.get('failed_adoptions', 0)})"
                )
        if len(summary.playbook_impact) > len(top_entries):
            remaining = len(summary.playbook_impact) - len(top_entries)
            lines.append(f"- … plus {remaining} additional entries documented in JSON output.")
    if summary.usage_metrics:
        usage = summary.usage_metrics
        lines.append("")
        lines.append("## Runtime Usage")
        lines.append(
            f"- Cue triggers: {usage.cue_triggers} across {usage.cue_trigger_sessions} sessions (rate: {_format_float(usage.cue_trigger_rate)})"
        )
        lines.append(
            f"- Action adoptions: {usage.adoption_events} (successful: {usage.successful_adoptions}, adoption rate: {_format_float(usage.adoption_rate)})"
        )
        lines.append(f"- Unique cue steps fired: {usage.unique_cue_steps}")
    if summary.efficiency:
        eff = summary.efficiency
        lines.append("")
        lines.append("## Efficiency Snapshot")
        lines.append(
            f"- Sessions with cues: {eff.sessions_with_cues}, without cues: {eff.sessions_without_cues}"
        )
        lines.append(
            "- Avg reward with cues / without cues: {0} / {1}".format(
                _format_float(eff.average_reward_with_cues),
                _format_float(eff.average_reward_without_cues),
            )
        )
        lines.append(
            "- Avg tokens with cues / without cues: {0} / {1}".format(
                _format_float(eff.average_tokens_with_cues),
                _format_float(eff.average_tokens_without_cues),
            )
        )
        if eff.reward_delta is not None:
            lines.append(f"- Reward delta (with - without cues): {_format_float(eff.reward_delta)}")
        if eff.token_delta is not None:
            lines.append(f"- Token delta (with - without cues): {_format_float(eff.token_delta)}")
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


def _build_playbook_metrics(playbook_meta: dict[str, Any] | None) -> PlaybookMetricsSummary | None:
    if not isinstance(playbook_meta, dict) or not playbook_meta:
        return None
    summary = _coerce_dict(playbook_meta.get("playbook_summary"))
    if not summary:
        return None
    total = int(summary.get("total_candidates") or 0)
    passed = int(summary.get("passed") or 0)
    failed = int(summary.get("failed") or 0)
    pass_rate = _coerce_float(summary.get("pass_rate"))
    avg_weighted = _coerce_float(summary.get("average_weighted_score"))
    gate_failures = summary.get("gate_failures") if isinstance(summary.get("gate_failures"), dict) else {}
    weights = summary.get("weights") if isinstance(summary.get("weights"), dict) else None
    return PlaybookMetricsSummary(
        total_candidates=total,
        passed=passed,
        failed=failed,
        pass_rate=pass_rate,
        gate_failures=gate_failures,
        average_weighted_score=avg_weighted,
        weights=weights,
    )


def _build_playbook_lifecycle_summary(
    playbook_meta: dict[str, Any] | None,
    rejected_count: int,
) -> PlaybookLifecycleSummary | None:
    if not isinstance(playbook_meta, dict):
        return None
    reinforcement_active = reinforcement_deprecated = 0
    differentiation_active = differentiation_deprecated = 0
    entries = playbook_meta.get("playbook_entries")
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            provenance = entry.get("provenance") if isinstance(entry.get("provenance"), dict) else {}
            status = provenance.get("status") if isinstance(provenance.get("status"), dict) else {}
            category = (status.get("category") or "").lower()
            lifecycle = (status.get("lifecycle") or "").lower()
            if category == "reinforcement":
                if lifecycle == "active":
                    reinforcement_active += 1
                elif lifecycle == "deprecated":
                    reinforcement_deprecated += 1
            elif category == "differentiation":
                if lifecycle == "active":
                    differentiation_active += 1
                elif lifecycle == "deprecated":
                    differentiation_deprecated += 1
    if not (reinforcement_active or reinforcement_deprecated or differentiation_active or differentiation_deprecated or rejected_count):
        return None
    return PlaybookLifecycleSummary(
        reinforcement_active=reinforcement_active,
        reinforcement_deprecated=reinforcement_deprecated,
        differentiation_active=differentiation_active,
        differentiation_deprecated=differentiation_deprecated,
        rejected=rejected_count,
    )


def _build_playbook_impact(playbook_meta: dict[str, Any] | None) -> list[PlaybookImpactEntry]:
    if not isinstance(playbook_meta, dict):
        return []
    entries = playbook_meta.get("playbook_entries")
    if not isinstance(entries, list):
        return []
    impact_entries: list[PlaybookImpactEntry] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        impact = entry.get("impact")
        if not isinstance(impact, dict):
            continue
        entry_id = entry.get("id") or ""
        audience = entry.get("audience")
        cue = entry.get("cue") if isinstance(entry.get("cue"), dict) else {}
        action = entry.get("action") if isinstance(entry.get("action"), dict) else {}
        total_cue_hits = int(impact.get("total_cue_hits") or 0)
        adoption_events = int(impact.get("total_adoptions") or 0)
        successful_adoptions = int(impact.get("successful_adoptions") or 0)
        failed_adoptions = int(impact.get("failed_adoptions") or 0)
        adoption_rate = _safe_ratio(successful_adoptions, total_cue_hits)
        avg_reward_with = _safe_ratio(impact.get("reward_with_sum"), impact.get("reward_with_count"))
        avg_reward_without = _safe_ratio(impact.get("reward_without_sum"), impact.get("reward_without_count"))
        reward_delta = None
        if avg_reward_with is not None and avg_reward_without is not None:
            reward_delta = avg_reward_with - avg_reward_without
        avg_tokens_with = _safe_ratio(impact.get("tokens_with_sum"), impact.get("tokens_with_count"))
        avg_tokens_without = _safe_ratio(impact.get("tokens_without_sum"), impact.get("tokens_without_count"))
        token_delta = None
        if avg_tokens_with is not None and avg_tokens_without is not None:
            token_delta = avg_tokens_with - avg_tokens_without
        incident_ids = []
        raw_incidents = impact.get("incident_ids")
        if isinstance(raw_incidents, list):
            incident_ids = [str(item) for item in raw_incidents if isinstance(item, str)]
        unique_incidents = len(set(incident_ids))
        transfer_success = unique_incidents > 1
        transfer_level = (
            "universal" if unique_incidents >= 10 else
            "workflow" if unique_incidents >= 5 else
            "domain" if unique_incidents >= 2 else
            "task"
        )
        retry_avg = _safe_ratio(impact.get("retry_sum"), impact.get("retry_samples"))
        failure_events = int(impact.get("failure_events") or 0)
        failure_stats = None
        if retry_avg is not None or failure_events:
            failure_stats = {
                "retry_avg": retry_avg,
                "retry_samples": int(impact.get("retry_samples") or 0),
                "failure_events": failure_events,
                "failed_adoptions": failed_adoptions,
            }
        impact_score = None
        if adoption_rate is not None and reward_delta is not None:
            impact_score = adoption_rate * reward_delta
        impact_entries.append(
            PlaybookImpactEntry(
                entry_id=str(entry_id),
                audience=audience if isinstance(audience, str) else None,
                cue_pattern=cue.get("pattern") if isinstance(cue.get("pattern"), str) else None,
                runtime_handle=action.get("runtime_handle") if isinstance(action.get("runtime_handle"), str) else None,
                total_cue_hits=total_cue_hits,
                adoption_events=adoption_events,
                successful_adoptions=successful_adoptions,
                adoption_rate=adoption_rate,
                average_reward_with=avg_reward_with,
                average_reward_without=avg_reward_without,
                reward_delta=reward_delta,
                average_tokens_with=avg_tokens_with,
                average_tokens_without=avg_tokens_without,
                token_delta=token_delta,
                unique_incidents=unique_incidents,
                transfer_success=transfer_success,
                transfer_level=transfer_level,
                failure_avoidance=failure_stats,
                impact_score=impact_score,
            )
        )
    impact_entries.sort(
        key=lambda item: (
            _sort_desc(item.impact_score),
            _sort_desc(item.reward_delta),
            item.entry_id,
        )
    )
    return impact_entries


def _build_usage_metrics(cue_stats: dict[str, int], total_sessions: int) -> UsageMetrics | None:
    if total_sessions <= 0:
        return None
    cue_triggers = int(cue_stats.get("cue_triggers", 0))
    cue_sessions = int(cue_stats.get("cue_sessions", 0))
    unique_cue_steps = int(cue_stats.get("unique_cue_steps", 0))
    adoptions = int(cue_stats.get("adoptions", 0))
    success = int(cue_stats.get("success", 0))
    if not (cue_triggers or adoptions or success or cue_sessions):
        return None
    adoption_rate = (success / adoptions) if adoptions else None
    cue_trigger_rate = (cue_sessions / total_sessions) if total_sessions and cue_sessions else None
    return UsageMetrics(
        cue_triggers=cue_triggers,
        cue_trigger_sessions=cue_sessions,
        unique_cue_steps=unique_cue_steps,
        adoption_events=adoptions,
        successful_adoptions=success,
        adoption_rate=adoption_rate,
        cue_trigger_rate=cue_trigger_rate,
    )


def _build_efficiency_snapshot(
    reward_with_cues: Sequence[float],
    reward_without_cues: Sequence[float],
    tokens_with_cues: Sequence[float],
    tokens_without_cues: Sequence[float],
) -> EfficiencySnapshot | None:
    has_rewards = bool(reward_with_cues or reward_without_cues)
    has_tokens = bool(tokens_with_cues or tokens_without_cues)
    if not has_rewards and not has_tokens:
        return None
    avg_reward_with = fmean(reward_with_cues) if reward_with_cues else None
    avg_reward_without = fmean(reward_without_cues) if reward_without_cues else None
    avg_tokens_with = fmean(tokens_with_cues) if tokens_with_cues else None
    avg_tokens_without = fmean(tokens_without_cues) if tokens_without_cues else None
    reward_delta = None
    if avg_reward_with is not None and avg_reward_without is not None:
        reward_delta = avg_reward_with - avg_reward_without
    token_delta = None
    if avg_tokens_with is not None and avg_tokens_without is not None:
        token_delta = avg_tokens_with - avg_tokens_without
    return EfficiencySnapshot(
        sessions_with_cues=len(reward_with_cues) if reward_with_cues else 0,
        sessions_without_cues=len(reward_without_cues) if reward_without_cues else 0,
        average_reward_with_cues=avg_reward_with,
        average_reward_without_cues=avg_reward_without,
        average_tokens_with_cues=avg_tokens_with,
        average_tokens_without_cues=avg_tokens_without,
        reward_delta=reward_delta,
        token_delta=token_delta,
    )


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


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    numerator_value = _coerce_float(numerator)
    denominator_value = _coerce_float(denominator)
    if numerator_value is None or denominator_value is None:
        return None
    if denominator_value == 0:
        return None
    return numerator_value / denominator_value


def _sort_desc(value: float | None) -> float:
    if value is None:
        return float("inf")
    return -value


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


def _format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"
