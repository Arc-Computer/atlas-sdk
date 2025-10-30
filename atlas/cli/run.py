"""Runtime helper command consuming discovery metadata."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
import warnings
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

warnings.filterwarnings(
    "ignore",
    message='Field name "schema" in "LearningConfig" shadows an attribute in parent "BaseModel"',
    category=UserWarning,
)

from atlas.cli.utils import CLIError, execute_runtime, parse_env_flags, write_run_record
from atlas.core import arun as atlas_arun
from atlas.runtime.models import IntermediateStep, IntermediateStepType
from atlas.runtime.orchestration.execution_context import ExecutionContext
from atlas.utils.env import load_dotenv_if_available
from atlas.sdk.discovery import calculate_file_hash


DISCOVERY_FILENAME = "discover.json"


def _load_metadata(path: Path) -> dict[str, object]:
    if not path.exists():
        raise CLIError(f"Discovery metadata not found at {path}. Run `atlas env init` first.")
    try:
        raw = path.read_text(encoding="utf-8")
        metadata = json.loads(raw)
    except Exception as exc:
        raise CLIError(f"Failed to load discovery metadata: {exc}") from exc
    if not isinstance(metadata, dict):
        raise CLIError("Discovery metadata is malformed.")
    return metadata


def _validate_module_hash(project_root: Path, payload: dict[str, object], role: str) -> None:
    expected_hash = payload.get("hash")
    rel_path = payload.get("file")
    module = payload.get("module")
    if not expected_hash or not rel_path:
        return
    if not isinstance(expected_hash, str) or not isinstance(rel_path, str):
        raise CLIError(f"Discovery metadata missing hash for {role}. Re-run `atlas env init`.")
    file_path = project_root / rel_path
    if not file_path.exists():
        raise CLIError(f"{role.title()} module '{module}' not found at {file_path}. Re-run `atlas env init`.")
    current_hash = calculate_file_hash(file_path)
    if current_hash != expected_hash:
        raise CLIError(
            f"{role.title()} module '{module}' has changed since discovery. "
            "Run `atlas env init` again to refresh metadata."
        )


def _ensure_jsonable(value: Any, depth: int = 0) -> Any:
    if depth > 10:
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _ensure_jsonable(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_ensure_jsonable(item, depth + 1) for item in value]
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
        except Exception:
            return str(value)
        return _ensure_jsonable(dumped, depth + 1)
    if hasattr(value, "to_dict"):
        try:
            dumped = value.to_dict()
        except Exception:
            return str(value)
        return _ensure_jsonable(dumped, depth + 1)
    if hasattr(value, "__dict__"):
        return _ensure_jsonable(vars(value), depth + 1)
    return str(value)


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _clean_text(value: Any, *, limit: int = 240) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    text = value.strip()
    if not text:
        return None
    if len(text) > limit:
        return text[: limit - 3].rstrip() + "..."
    return text


def _extract_reward_stats(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "score": None,
        "recent_mean": None,
        "recent_count": None,
        "baseline_mean": None,
        "baseline_count": None,
        "delta": None,
    }
    if not isinstance(metadata, Mapping):
        return snapshot
    reward_summary = metadata.get("reward_summary")
    if isinstance(reward_summary, Mapping):
        snapshot["score"] = _coerce_float(reward_summary.get("score"))
        recent_mean = reward_summary.get("recent_mean")
        if recent_mean is not None:
            snapshot["recent_mean"] = _coerce_float(recent_mean)
        baseline_mean = reward_summary.get("baseline_mean")
        if baseline_mean is not None:
            snapshot["baseline_mean"] = _coerce_float(baseline_mean)
        recent_count = reward_summary.get("recent_count")
        if recent_count is not None:
            snapshot["recent_count"] = _coerce_int(recent_count)
        baseline_count = reward_summary.get("baseline_count")
        if baseline_count is not None:
            snapshot["baseline_count"] = _coerce_int(baseline_count)
    session_reward = metadata.get("session_reward")
    if snapshot["score"] is None and isinstance(session_reward, Mapping):
        snapshot["score"] = _coerce_float(session_reward.get("score"))
    reward_stats = metadata.get("session_reward_stats")
    if isinstance(reward_stats, Mapping):
        if snapshot["score"] is None:
            snapshot["score"] = _coerce_float(reward_stats.get("score"))
        if snapshot["recent_mean"] is None:
            snapshot["recent_mean"] = _coerce_float(
                reward_stats.get("recent_mean")
                or reward_stats.get("rolling_mean")
                or reward_stats.get("score_mean")
            )
        if snapshot["recent_count"] is None:
            snapshot["recent_count"] = _coerce_int(
                reward_stats.get("recent_count")
                or reward_stats.get("recent_window")
                or reward_stats.get("recent_sample_count")
            )
        if snapshot["baseline_mean"] is None:
            snapshot["baseline_mean"] = _coerce_float(reward_stats.get("baseline_mean"))
        if snapshot["baseline_count"] is None:
            snapshot["baseline_count"] = _coerce_int(reward_stats.get("baseline_count"))
        baseline_block = reward_stats.get("baseline")
        if isinstance(baseline_block, Mapping):
            if snapshot["baseline_mean"] is None:
                snapshot["baseline_mean"] = _coerce_float(
                    baseline_block.get("score_mean") or baseline_block.get("score")
                )
            if snapshot["baseline_count"] is None:
                snapshot["baseline_count"] = _coerce_int(baseline_block.get("sample_count"))
        if snapshot["baseline_count"] is None:
            snapshot["baseline_count"] = _coerce_int(reward_stats.get("sample_count"))
    drift_payload = metadata.get("drift")
    if isinstance(drift_payload, Mapping):
        baseline = drift_payload.get("baseline")
        if isinstance(baseline, Mapping):
            if snapshot["baseline_mean"] is None:
                snapshot["baseline_mean"] = _coerce_float(baseline.get("score_mean"))
            if snapshot["baseline_count"] is None:
                snapshot["baseline_count"] = _coerce_int(baseline.get("sample_count"))
    recent_mean = snapshot["recent_mean"]
    baseline_mean = snapshot["baseline_mean"]
    if recent_mean is not None and baseline_mean is not None:
        snapshot["delta"] = recent_mean - baseline_mean
    return snapshot


def _format_reward_summary_line(metadata: Mapping[str, Any]) -> str:
    stats = _extract_reward_stats(metadata)
    parts: list[str] = []
    score = stats.get("score")
    if isinstance(score, (int, float)):
        parts.append(f"latest={score:.2f}")
    recent_mean = stats.get("recent_mean")
    if isinstance(recent_mean, (int, float)):
        label = f"recent={recent_mean:.2f}"
        recent_count = stats.get("recent_count")
        if isinstance(recent_count, int) and recent_count > 0:
            label += f" (n={recent_count})"
        parts.append(label)
    baseline_mean = stats.get("baseline_mean")
    if isinstance(baseline_mean, (int, float)):
        label = f"baseline={baseline_mean:.2f}"
        baseline_count = stats.get("baseline_count")
        if isinstance(baseline_count, int) and baseline_count > 0:
            label += f" (n={baseline_count})"
        parts.append(label)
    delta = stats.get("delta")
    if isinstance(delta, (int, float)):
        parts.append(f"Δ={delta:+.2f}")
    return "; ".join(parts)


def _extract_token_usage(metadata: Mapping[str, Any]) -> Mapping[str, Any] | None:
    usage = metadata.get("token_usage")
    if isinstance(usage, Mapping):
        return usage
    learning_usage = metadata.get("learning_usage")
    if isinstance(learning_usage, Mapping):
        session_block = learning_usage.get("session")
        if isinstance(session_block, Mapping):
            tokens = session_block.get("token_usage")
            if isinstance(tokens, Mapping):
                return tokens
    return None


def _format_token_usage_line(metadata: Mapping[str, Any]) -> str:
    usage = _extract_token_usage(metadata)
    if not isinstance(usage, Mapping):
        return ""
    prompt = _coerce_int(usage.get("prompt_tokens"))
    completion = _coerce_int(usage.get("completion_tokens"))
    total = _coerce_int(usage.get("total_tokens"))
    calls = _coerce_int(usage.get("calls"))
    parts: list[str] = []
    if prompt is not None:
        parts.append(f"prompt={prompt}")
    if completion is not None:
        parts.append(f"completion={completion}")
    if total is not None:
        parts.append(f"total={total}")
    if calls is not None:
        parts.append(f"calls={calls}")
    return " ".join(parts)


def _format_usage_summary_line(metadata: Mapping[str, Any]) -> str:
    usage = metadata.get("learning_usage")
    if not isinstance(usage, Mapping):
        return ""
    session_block = usage.get("session")
    if not isinstance(session_block, Mapping):
        return ""
    cue_hits = _coerce_int(session_block.get("cue_hits"))
    adoption_count = _coerce_int(session_block.get("action_adoptions"))
    failed_adoptions = _coerce_int(session_block.get("failed_adoptions")) or 0
    success_adoptions = None
    if adoption_count is not None:
        success_adoptions = adoption_count - failed_adoptions
        if success_adoptions < 0:
            success_adoptions = 0
    unique_cue_steps = session_block.get("unique_cue_steps")
    unique_adoption_steps = session_block.get("unique_adoption_steps")
    parts: list[str] = []
    if cue_hits is not None:
        segment = f"cue_hits={cue_hits}"
        if isinstance(unique_cue_steps, Sequence):
            segment += f" (steps={len(unique_cue_steps)})"
        parts.append(segment)
    if adoption_count is not None:
        segment = f"adoptions={adoption_count}"
        details: list[str] = []
        if success_adoptions is not None:
            details.append(f"success={success_adoptions}")
        if failed_adoptions:
            details.append(f"failed={failed_adoptions}")
        if details:
            segment += f" ({', '.join(details)})"
        if isinstance(unique_adoption_steps, Sequence):
            segment += f" steps={len(unique_adoption_steps)}"
        parts.append(segment)
    return "; ".join(parts)


def _collect_learning_notes(metadata: Mapping[str, Any]) -> list[str]:
    notes: list[str] = []
    student_note = _clean_text(metadata.get("session_student_learning"))
    if student_note:
        notes.append(f"Student: {student_note}")
    teacher_note = _clean_text(metadata.get("session_teacher_learning"))
    if teacher_note:
        notes.append(f"Teacher: {teacher_note}")
    session_note = _clean_text(metadata.get("session_learning_note"))
    if session_note:
        notes.append(f"Note: {session_note}")
    teacher_notes = metadata.get("teacher_notes")
    if isinstance(teacher_notes, Sequence):
        cleaned = [_clean_text(entry) for entry in teacher_notes]
        cleaned = [entry for entry in cleaned if entry]
        if cleaned:
            excerpt = "; ".join(cleaned[:2])
            notes.append(f"Teacher Notes: {excerpt}")
    return notes


def _average(impact: Mapping[str, Any] | None, sum_key: str, count_key: str) -> float | None:
    if not isinstance(impact, Mapping):
        return None
    total_sum = _coerce_float(impact.get(sum_key))
    count = _coerce_int(impact.get(count_key))
    if total_sum is None or not count:
        return None
    return total_sum / count


def _active_playbook_lines(metadata: Mapping[str, Any]) -> list[str]:
    state = metadata.get("learning_state")
    if not isinstance(state, Mapping):
        return []
    meta = state.get("metadata")
    if not isinstance(meta, Mapping):
        return []
    entries = meta.get("playbook_entries")
    if not isinstance(entries, Sequence):
        return []
    active_entries = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        provenance = entry.get("provenance")
        lifecycle = None
        if isinstance(provenance, Mapping):
            status_block = provenance.get("status")
            if isinstance(status_block, Mapping):
                lifecycle = status_block.get("lifecycle")
        if lifecycle != "active":
            continue
        active_entries.append(entry)
    lines: list[str] = []
    for entry in sorted(active_entries, key=lambda item: str(item.get("id") or "")):
        entry_id = str(entry.get("id") or "entry")
        action = entry.get("action") if isinstance(entry.get("action"), Mapping) else None
        runtime_handle = action.get("runtime_handle") if isinstance(action, Mapping) else None
        impact = entry.get("impact") if isinstance(entry.get("impact"), Mapping) else {}
        cue_hits = _coerce_int(impact.get("total_cue_hits"))
        adoption_total = _coerce_int(impact.get("total_adoptions"))
        success_total = _coerce_int(impact.get("successful_adoptions"))
        failed_total = _coerce_int(impact.get("failed_adoptions"))
        if success_total is None and adoption_total is not None and failed_total is not None:
            success_total = max(adoption_total - failed_total, 0)
        reward_with = _average(impact, "reward_with_sum", "reward_with_count")
        reward_without = _average(impact, "reward_without_sum", "reward_without_count")
        reward_delta = None
        if reward_with is not None and reward_without is not None:
            reward_delta = reward_with - reward_without
        token_with = _average(impact, "tokens_with_sum", "tokens_with_count")
        token_without = _average(impact, "tokens_without_sum", "tokens_without_count")
        token_delta = None
        if token_with is not None and token_without is not None:
            token_delta = token_with - token_without
        label = entry_id if not runtime_handle else f"{entry_id} [{runtime_handle}]"
        metrics: list[str] = []
        if cue_hits is not None:
            metrics.append(f"cue_hits={cue_hits}")
        if adoption_total is not None:
            if success_total is not None:
                metrics.append(f"adoptions={success_total}/{adoption_total}")
            else:
                metrics.append(f"adoptions={adoption_total}")
        if reward_with is not None:
            reward_line = f"reward_with={reward_with:.2f}"
            if reward_without is not None:
                reward_line += f" without={reward_without:.2f}"
            if reward_delta is not None:
                reward_line += f" (Δ{reward_delta:+.2f})"
            metrics.append(reward_line)
        elif reward_without is not None:
            metrics.append(f"reward_without={reward_without:.2f}")
        if token_delta is not None and abs(token_delta) >= 50:
            token_parts: list[str] = []
            if token_with is not None:
                token_parts.append(f"with={token_with:.0f}")
            if token_without is not None:
                token_parts.append(f"without={token_without:.0f}")
            token_line = "tokens_" + "/".join(token_parts)
            token_line += f" (Δ{token_delta:+.0f})"
            metrics.append(token_line)
        if metrics:
            lines.append(f"{label} — {'; '.join(metrics)}")
        else:
            lines.append(label)
    return lines


def _format_recent_failures_line(metadata: Mapping[str, Any]) -> str | None:
    state = metadata.get("learning_state")
    if not isinstance(state, Mapping):
        return None
    meta = state.get("metadata")
    if not isinstance(meta, Mapping):
        return None
    failure_block = meta.get("last_failure")
    if not isinstance(failure_block, Mapping):
        return None
    candidates = failure_block.get("rejected_candidates")
    if not isinstance(candidates, Sequence) or not candidates:
        return None
    timestamp = failure_block.get("timestamp")
    timestamp_str = _clean_text(timestamp, limit=24) if timestamp else "unknown"
    return f"rejected={len(candidates)} (latest: {timestamp_str})"


def _extract_learning_key(metadata: Mapping[str, Any]) -> str | None:
    learning_key = metadata.get("learning_key")
    if isinstance(learning_key, str) and learning_key.strip():
        return learning_key.strip()
    session_meta = metadata.get("session_metadata")
    if isinstance(session_meta, Mapping):
        key = session_meta.get("learning_key")
        if isinstance(key, str) and key.strip():
            return key.strip()
    return None


def _render_learning_summary(metadata: Mapping[str, Any] | None, *, stream: bool = False) -> str:
    if not isinstance(metadata, Mapping):
        return ""
    lines: list[str] = []
    heading = "=== Learning Summary ===" if not stream else "--- Learning Summary ---"
    lines.append(heading)
    reward_line = _format_reward_summary_line(metadata)
    if reward_line:
        lines.append(f"Reward: {reward_line}")
    token_line = _format_token_usage_line(metadata)
    if token_line:
        lines.append(f"Tokens: {token_line}")
    usage_line = _format_usage_summary_line(metadata)
    if usage_line:
        lines.append(f"Usage: {usage_line}")
    notes = _collect_learning_notes(metadata)
    if notes:
        lines.append("Learning Notes:")
        lines.extend(f"  {note}" for note in notes)
    playbook_lines = _active_playbook_lines(metadata)
    if playbook_lines:
        lines.append("Active Playbook Entries:")
        lines.extend(f"  {line}" for line in playbook_lines)
    failure_summary = _format_recent_failures_line(metadata)
    if failure_summary:
        lines.append(f"Recent Failures: {failure_summary}")
    learning_key = _extract_learning_key(metadata)
    if learning_key:
        lines.append(f"Learning Key: {learning_key}")
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def _snapshot_learning_usage(metadata: Mapping[str, Any]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"roles": {}, "session": {"cue_hits": 0, "action_adoptions": 0, "failed_adoptions": 0}}
    if not isinstance(metadata, Mapping):
        return snapshot
    usage = metadata.get("learning_usage")
    if not isinstance(usage, Mapping):
        return snapshot
    session_block = usage.get("session")
    if isinstance(session_block, Mapping):
        session_snapshot: dict[str, Any] = {
            "cue_hits": _coerce_int(session_block.get("cue_hits")) or 0,
            "action_adoptions": _coerce_int(session_block.get("action_adoptions")) or 0,
            "failed_adoptions": _coerce_int(session_block.get("failed_adoptions")) or 0,
        }
        token_usage = session_block.get("token_usage")
        if isinstance(token_usage, Mapping):
            session_snapshot["token_usage"] = {
                "prompt_tokens": _coerce_int(token_usage.get("prompt_tokens")) or 0,
                "completion_tokens": _coerce_int(token_usage.get("completion_tokens")) or 0,
                "total_tokens": _coerce_int(token_usage.get("total_tokens")) or 0,
                "calls": _coerce_int(token_usage.get("calls")) or 0,
            }
        unique_cue_steps = session_block.get("unique_cue_steps")
        if isinstance(unique_cue_steps, Sequence):
            session_snapshot["unique_cue_steps"] = len(unique_cue_steps)
        unique_adoption_steps = session_block.get("unique_adoption_steps")
        if isinstance(unique_adoption_steps, Sequence):
            session_snapshot["unique_adoption_steps"] = len(unique_adoption_steps)
        snapshot["session"] = session_snapshot
    roles_store = usage.get("roles")
    if isinstance(roles_store, Mapping):
        for role, entries in roles_store.items():
            if not isinstance(entries, Mapping):
                continue
            role_snapshot: dict[str, dict[str, Any]] = {}
            for entry_id, details in entries.items():
                if not isinstance(details, Mapping):
                    continue
                metadata_block = details.get("metadata") if isinstance(details.get("metadata"), Mapping) else None
                category = None
                if isinstance(metadata_block, Mapping):
                    scope_block = metadata_block.get("scope")
                    if isinstance(scope_block, Mapping):
                        category = scope_block.get("category")
                role_snapshot[str(entry_id)] = {
                    "cue_hits": _coerce_int(details.get("cue_hits")) or 0,
                    "action_adoptions": _coerce_int(details.get("action_adoptions")) or 0,
                    "successful_adoptions": _coerce_int(details.get("successful_adoptions")) or 0,
                    "failed_adoptions": _coerce_int(details.get("failed_adoptions")) or 0,
                    "runtime_handle": details.get("runtime_handle"),
                    "category": category,
                }
            if role_snapshot:
                snapshot["roles"][str(role)] = role_snapshot
    return snapshot


def _snapshot_token_usage(metadata: Mapping[str, Any]) -> dict[str, int]:
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
    }
    usage = _extract_token_usage(metadata)
    if isinstance(usage, Mapping):
        for key in totals.keys():
            value = _coerce_int(usage.get(key))
            if value is not None:
                totals[key] = value
        return totals
    return totals


class _StreamMonitor:
    """Render live execution events while the orchestrator runs."""

    # NOTE: This helper interacts with the real ExecutionContext event stream and is
    # therefore exercised via end-to-end runs rather than unit tests.
    def __init__(self, context: ExecutionContext) -> None:
        self._context = context
        self._usage_snapshot = _snapshot_learning_usage(context.metadata)
        self._token_snapshot = _snapshot_token_usage(context.metadata)
        self._attempt_meta: dict[str, dict[str, Any]] = {}

    def handle_event(self, event: IntermediateStep) -> None:
        prefix = f"[{datetime.now().strftime('%H:%M:%S')}]"
        if event.event_type == IntermediateStepType.TASK_START:
            self._handle_task_start(event, prefix)
        elif event.event_type == IntermediateStepType.TASK_END:
            self._handle_task_end(event, prefix)
        self._emit_usage_changes(prefix)
        self._emit_token_changes(prefix)

    def _handle_task_start(self, event: IntermediateStep, prefix: str) -> None:
        data = event.payload.data
        payload = data.input if data is not None else None
        step_payload = payload.get("step") if isinstance(payload, Mapping) else None
        attempt = _coerce_int(payload.get("attempt")) if isinstance(payload, Mapping) else None
        step_id = step_payload.get("id") if isinstance(step_payload, Mapping) else None
        description = _clean_text(step_payload.get("description"), limit=120) if isinstance(step_payload, Mapping) else None
        tool = step_payload.get("tool") if isinstance(step_payload, Mapping) else None
        attempt_id = event.payload.UUID
        self._attempt_meta[attempt_id] = {
            "step_id": step_id,
            "description": description,
            "tool": tool,
            "attempt": attempt,
        }
        label_parts: list[str] = []
        label_parts.append(f"step {step_id}" if step_id is not None else "step")
        if attempt:
            label_parts.append(f"(attempt {attempt})")
        label = " ".join(label_parts)
        detail_parts: list[str] = []
        if description:
            detail_parts.append(description)
        if tool:
            detail_parts.append(f"tool: {tool}")
        detail = " — ".join(detail_parts) if detail_parts else "starting"
        print(f"{prefix} {label} ▶ {detail}")

    def _handle_task_end(self, event: IntermediateStep, prefix: str) -> None:
        attempt_id = event.payload.UUID
        attempt_meta = self._attempt_meta.pop(attempt_id, {})
        step_id = attempt_meta.get("step_id")
        description = attempt_meta.get("description")
        attempt = attempt_meta.get("attempt")
        data = event.payload.data
        payload = data.output if data is not None else None
        evaluation = payload.get("evaluation") if isinstance(payload, Mapping) else None
        status = payload.get("status") if isinstance(payload, Mapping) else None
        runtime_block = payload.get("runtime") if isinstance(payload, Mapping) else None
        duration_ms = None
        if isinstance(runtime_block, Mapping):
            timings = runtime_block.get("timings_ms")
            if isinstance(timings, Mapping):
                duration_ms = _coerce_float(timings.get("total_ms"))
        reward_score = None
        uncertainty = None
        if isinstance(evaluation, Mapping):
            reward_block = evaluation.get("reward")
            if isinstance(reward_block, Mapping):
                reward_score = _coerce_float(reward_block.get("score"))
                raw_block = reward_block.get("raw")
                if isinstance(raw_block, Mapping):
                    uncertainty = _coerce_float(
                        raw_block.get("uncertainty")
                        or raw_block.get("uncertainty_mean")
                        or raw_block.get("reward_uncertainty")
                    )
        label_parts: list[str] = []
        label_parts.append(f"step {step_id}" if step_id is not None else "step")
        if attempt:
            label_parts.append(f"(attempt {attempt})")
        label = " ".join(label_parts)
        summary_parts: list[str] = []
        if reward_score is not None:
            summary_parts.append(f"reward={reward_score:.2f}")
        if uncertainty is not None:
            summary_parts.append(f"uncertainty={uncertainty:.2f}")
        if duration_ms is not None:
            summary_parts.append(f"{duration_ms:.0f} ms")
        if status:
            status_text = str(status).lower()
            if status_text not in {"passed", "completed"}:
                summary_parts.append(f"status={status}")
        summary = " ".join(summary_parts) if summary_parts else "completed"
        if description:
            summary = f"{summary} — {description}"
        print(f"{prefix} {label} ✓ {summary}")

    def _emit_usage_changes(self, prefix: str) -> None:
        current = _snapshot_learning_usage(self._context.metadata)
        previous = self._usage_snapshot
        current_roles = current.get("roles", {}) if isinstance(current, Mapping) else {}
        previous_roles = previous.get("roles", {}) if isinstance(previous, Mapping) else {}
        for role, entries in current_roles.items():
            prev_entries = previous_roles.get(role, {}) if isinstance(previous_roles, Mapping) else {}
            for entry_id, stats in entries.items():
                prev_stats = prev_entries.get(entry_id, {}) if isinstance(prev_entries, Mapping) else {}
                delta_hits = stats.get("cue_hits", 0) - prev_stats.get("cue_hits", 0)
                if delta_hits > 0:
                    handle = stats.get("runtime_handle")
                    handle_text = f" [{handle}]" if handle else ""
                    print(f"{prefix} cue {entry_id}{handle_text} +{delta_hits} (total {stats.get('cue_hits', 0)})")
                delta_adopt = stats.get("action_adoptions", 0) - prev_stats.get("action_adoptions", 0)
                if delta_adopt > 0:
                    handle = stats.get("runtime_handle")
                    handle_text = f" [{handle}]" if handle else ""
                    success_delta = stats.get("successful_adoptions", 0) - prev_stats.get("successful_adoptions", 0)
                    fail_delta = stats.get("failed_adoptions", 0) - prev_stats.get("failed_adoptions", 0)
                    fragments: list[str] = []
                    if success_delta > 0:
                        fragments.append(f"success +{success_delta}")
                    if fail_delta > 0:
                        fragments.append(f"fail +{fail_delta}")
                    if not fragments:
                        fragments.append(f"+{delta_adopt}")
                    success_total = stats.get("successful_adoptions", 0)
                    total_adoptions = stats.get("action_adoptions", 0)
                    fragment_text = ", ".join(fragments)
                    print(
                        f"{prefix} adoption {entry_id}{handle_text} {fragment_text} (total {success_total}/{total_adoptions})"
                    )
                    reward_stats = _extract_reward_stats(self._context.metadata)
                    delta_reward = reward_stats.get("delta")
                    recent_mean = reward_stats.get("recent_mean")
                    baseline_mean = reward_stats.get("baseline_mean")
                    if isinstance(delta_reward, (int, float)) and isinstance(recent_mean, (int, float)) and isinstance(
                        baseline_mean, (int, float)
                    ):
                        print(
                            f"{prefix} impact Δreward {delta_reward:+.2f} (recent {recent_mean:.2f} vs baseline {baseline_mean:.2f})"
                        )
        self._usage_snapshot = current

    def _emit_token_changes(self, prefix: str) -> None:
        current = _snapshot_token_usage(self._context.metadata)
        prev = self._token_snapshot
        delta_prompt = current.get("prompt_tokens", 0) - prev.get("prompt_tokens", 0)
        delta_completion = current.get("completion_tokens", 0) - prev.get("completion_tokens", 0)
        delta_total = current.get("total_tokens", 0) - prev.get("total_tokens", 0)
        delta_calls = current.get("calls", 0) - prev.get("calls", 0)
        if any(value > 0 for value in (delta_prompt, delta_completion, delta_total, delta_calls)):
            fragments: list[str] = []
            if delta_prompt > 0:
                fragments.append(f"prompt +{delta_prompt}")
            if delta_completion > 0:
                fragments.append(f"completion +{delta_completion}")
            if delta_total > 0:
                fragments.append(f"total +{delta_total}")
            if delta_calls > 0:
                fragments.append(f"calls +{delta_calls}")
            print(f"{prefix} tokens {'; '.join(fragments)}")
        self._token_snapshot = current

def _run_with_config(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        return 1
    project_root = Path(args.path or ".").resolve()
    atlas_dir = project_root / ".atlas"
    atlas_dir.mkdir(parents=True, exist_ok=True)
    load_dotenv_if_available(project_root / ".env")
    sys_path_candidates = [project_root]
    src_dir = project_root / "src"
    if src_dir.exists():
        sys_path_candidates.append(src_dir)
    for candidate in reversed(sys_path_candidates):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
    if getattr(args, "mode", None):
        print("[atlas run] --mode override is not yet supported; using configuration defaults.", file=sys.stderr)
    if getattr(args, "max_steps", None):
        print("[atlas run] --max-steps option is not currently supported.", file=sys.stderr)

    async def _invoke() -> tuple[Any | None, dict[str, Any]]:
        execution_context = ExecutionContext.get()
        execution_context.reset()
        stream_enabled = bool(getattr(args, "stream", False))
        monitor: _StreamMonitor | None = None
        queue: asyncio.Queue[IntermediateStep | None] | None = None
        consumer_task: asyncio.Task[None] | None = None

        def _event_handler(event: IntermediateStep) -> None:
            if queue is None:
                return
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:  # pragma: no cover - defensive guard
                pass

        if stream_enabled:
            monitor = _StreamMonitor(execution_context)
            queue = asyncio.Queue()

            async def _consume() -> None:
                try:
                    while True:
                        next_event = await queue.get()
                        if next_event is None:
                            break
                        if monitor is not None:
                            monitor.handle_event(next_event)
                except asyncio.CancelledError:
                    pass

            consumer_task = asyncio.create_task(_consume())

        result: Any | None = None
        try:
            result = await atlas_arun(
                args.task,
                str(config_path),
                stream_progress=stream_enabled,
                session_metadata={"source": "atlas run"},
                intermediate_step_handler=_event_handler if stream_enabled else None,
            )
        finally:
            if queue is not None:
                with contextlib.suppress(asyncio.QueueFull):
                    queue.put_nowait(None)
            if consumer_task is not None:
                with contextlib.suppress(asyncio.CancelledError):
                    await consumer_task

        metadata_snapshot: dict[str, Any] = dict(execution_context.metadata)
        return result, metadata_snapshot

    interrupted = False
    try:
        result, metadata_snapshot = asyncio.run(_invoke())
    except KeyboardInterrupt:
        interrupted = True
        metadata_snapshot = dict(ExecutionContext.get().metadata)
        result = None
    except Exception as exc:  # pragma: no cover - runtime failures surface to CLI
        print(f"Runtime worker failed: {exc}", file=sys.stderr)
        return 1

    metadata = _ensure_jsonable(metadata_snapshot)
    if getattr(args, "verbose", False):
        keys = list(metadata.keys()) if isinstance(metadata, dict) else []
        print(f"[atlas run] metadata keys captured: {keys}", file=sys.stderr)
    ExecutionContext.get().reset()

    run_payload = {
        "task": args.task,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "result": result.model_dump() if result is not None else None,
        "metadata": metadata,
    }
    run_path = write_run_record(atlas_dir, run_payload)

    if result is not None:
        final_answer = result.final_answer if isinstance(result.final_answer, str) else None
        if final_answer and final_answer.strip():
            print("\n=== Final Answer ===")
            print(final_answer.strip())
        else:
            print("\nNo final answer produced. Inspect telemetry for details.")
    elif interrupted:
        print("\nRun interrupted before the agent returned a final answer.")
    else:
        print("\nNo final answer produced. Inspect telemetry for details.")

    summary_text = _render_learning_summary(metadata if isinstance(metadata, Mapping) else {}, stream=bool(getattr(args, "stream", False)))
    if summary_text:
        print(f"\n{summary_text}")

    steps = metadata.get("steps") if isinstance(metadata, dict) else {}
    attempt_count = 0
    if isinstance(steps, dict):
        for entry in steps.values():
            if isinstance(entry, dict):
                attempts = entry.get("attempts", [])
                if isinstance(attempts, list):
                    attempt_count += len(attempts)
    print(f"\nTelemetry steps captured: {len(steps) if isinstance(steps, dict) else 0} (attempts={attempt_count})")
    print(f"Run artefact saved to {run_path}")
    return 130 if interrupted else 0


def _cmd_run(args: argparse.Namespace) -> int:
    if getattr(args, "config", None):
        return _run_with_config(args)
    if getattr(args, "stream", False):
        print("[atlas run] --stream flag is only supported with --config orchestrator runs; ignoring.", file=sys.stderr)
    project_root = Path(args.path or ".").resolve()
    atlas_dir = project_root / ".atlas"
    metadata_path = atlas_dir / DISCOVERY_FILENAME
    try:
        metadata = _load_metadata(metadata_path)
    except CLIError as exc:
        print(exc, file=sys.stderr)
        return 1
    project_root_value = metadata.get("project_root")
    if isinstance(project_root_value, Path):
        metadata_root = project_root_value.resolve()
    elif isinstance(project_root_value, str):
        metadata_root = Path(project_root_value).resolve()
    else:
        metadata_root = project_root
    env_payload = metadata.get("environment")
    agent_payload = metadata.get("agent")
    preflight = metadata.get("preflight")
    if not isinstance(env_payload, dict) or not isinstance(agent_payload, dict):
        print("Discovery metadata missing environment/agent payloads. Re-run `atlas env init`.", file=sys.stderr)
        return 1
    try:
        _validate_module_hash(metadata_root, env_payload, "environment")
        _validate_module_hash(metadata_root, agent_payload, "agent")
    except CLIError as exc:
        print(exc, file=sys.stderr)
        return 1
    try:
        env_overrides = parse_env_flags(args.env_vars or [])
    except CLIError as exc:
        print(exc, file=sys.stderr)
        return 1
    if isinstance(preflight, dict):
        notes = preflight.get("notes")
        if notes:
            print("Preflight notes from discovery:")
            for note in notes:
                print(f"  - {note}")
    capabilities_value = metadata.get("capabilities")
    capabilities: Dict[str, object] = capabilities_value if isinstance(capabilities_value, dict) else {}

    def _build_target(target_payload: dict[str, object]) -> tuple[dict[str, object] | None, dict[str, object] | None]:
        raw_kwargs = target_payload.get("kwargs")
        init_kwargs: dict[str, object] = dict(raw_kwargs) if isinstance(raw_kwargs, dict) else {}
        config_payload = target_payload.get("config")
        base_entry: dict[str, object] | None = None
        factory_entry: dict[str, object] | None = None
        module = target_payload.get("module")
        qualname = target_payload.get("qualname")
        module_str = module if isinstance(module, str) else None
        qualname_str = qualname if isinstance(qualname, str) else None
        if module_str and qualname_str:
            base_entry = {
                "module": module_str,
                "qualname": qualname_str,
            }
            if init_kwargs:
                base_entry["init_kwargs"] = init_kwargs
            if config_payload is not None:
                base_entry["config"] = config_payload
        factory_payload = target_payload.get("factory")
        if isinstance(factory_payload, dict):
            factory_module = factory_payload.get("module")
            factory_qualname = factory_payload.get("qualname")
            factory_kwargs = dict(init_kwargs)
            extra_kwargs = factory_payload.get("kwargs")
            if isinstance(extra_kwargs, dict):
                factory_kwargs.update(extra_kwargs)
            if isinstance(factory_module, str) and isinstance(factory_qualname, str):
                factory_entry = {
                    "module": factory_module,
                    "qualname": factory_qualname,
                    "kwargs": factory_kwargs,
                }
        return base_entry, factory_entry

    env_entry, env_factory_entry = _build_target(env_payload)
    agent_entry, agent_factory_entry = _build_target(agent_payload)

    spec: dict[str, object] = {
        "project_root": str(metadata_root),
        "task": args.task,
        "run_discovery": True,
        "env": env_overrides,
    }
    if env_entry:
        spec["environment"] = env_entry
    if env_factory_entry:
        spec["environment_factory"] = env_factory_entry
    if agent_entry:
        spec["agent"] = agent_entry
    if agent_factory_entry:
        spec["agent_factory"] = agent_factory_entry
    try:
        result, run_path = execute_runtime(
            spec,
            capabilities=capabilities,
            atlas_dir=atlas_dir,
            task=args.task,
            timeout=args.timeout or 300,
        )
    except CLIError as exc:
        print(f"Runtime worker failed: {exc}", file=sys.stderr)
        return 1
    final_answer = result.get("final_answer")
    if isinstance(final_answer, str) and final_answer.strip():
        print("\n=== Final Answer ===")
        print(final_answer.strip())
    else:
        print("\nNo final answer produced. Inspect telemetry for details.")
    telemetry_obj = result.get("telemetry")
    telemetry = telemetry_obj if isinstance(telemetry_obj, dict) else {}
    event_count = len(telemetry.get("events") or [])
    agent_emitted = telemetry.get("agent_emitted", False)
    print(f"\nTelemetry events captured: {event_count}")
    if not agent_emitted:
        print("Agent did not emit telemetry via emit_event; consider instrumenting emit_event calls.")
    print(f"Run artefact saved to {run_path}")
    return 0


def register_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    run_parser = subparsers.add_parser("run", help="Execute the discovered environment/agent pair.")
    run_parser.add_argument("--path", default=".", help="Project root containing .atlas/discover.json.")
    run_parser.add_argument(
        "--env",
        dest="env_vars",
        metavar="KEY=VALUE",
        action="append",
        default=[],
        help="Environment variable(s) to expose to the runtime worker.",
    )
    run_parser.add_argument(
        "--config",
        default=None,
        help="Path to an Atlas runtime configuration file. When provided, runs the full orchestrator stack instead of the discovery worker.",
    )
    run_parser.add_argument(
        "--mode",
        default=None,
        help="Requested execution mode override (experimental).",
    )
    run_parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of orchestrator steps (currently informational).",
    )
    run_parser.add_argument(
        "--task",
        required=True,
        help="Task prompt to send to the discovered agent.",
    )
    run_parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout (seconds) for the runtime worker (default: %(default)s).",
    )
    run_parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream intermediate execution updates while the run is in progress.",
    )
    run_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose CLI logging (includes metadata key summaries).",
    )
    run_parser.set_defaults(handler=_cmd_run)
