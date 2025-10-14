"""CLI utility for exporting persisted Atlas sessions as JSONL traces."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Optional, Protocol, Sequence

from atlas.config.models import StorageConfig
from atlas.runtime.storage.database import Database
from atlas.types import Plan, Step

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 100
DEFAULT_TRAJECTORY_LIMIT = 1000


class SessionStore(Protocol):
    """Subset of database operations required by the exporter."""

    async def fetch_sessions(self, limit: int = 50, offset: int = 0) -> List[dict[str, Any]]:
        raise NotImplementedError

    async def fetch_session(self, session_id: int) -> dict[str, Any] | None:
        raise NotImplementedError

    async def fetch_session_steps(self, session_id: int) -> List[dict[str, Any]]:
        raise NotImplementedError

    async def fetch_trajectory_events(self, session_id: int, limit: int = 200) -> List[dict[str, Any]]:
        raise NotImplementedError


@dataclass(slots=True)
class ExportSummary:
    sessions: int = 0
    steps: int = 0


# Backwards compatibility with earlier API name.
ExportStats = ExportSummary


@dataclass(slots=True)
class ExportRequest:
    database_url: str
    output_path: Path
    session_ids: Sequence[int] | None = None
    limit: int | None = None
    offset: int = 0
    status_filters: Sequence[str] | None = None
    trajectory_event_limit: int = DEFAULT_TRAJECTORY_LIMIT
    batch_size: int = DEFAULT_BATCH_SIZE
    min_connections: int = 1
    max_connections: int = 4
    statement_timeout_seconds: float = 30.0


async def export_sessions(
    store: SessionStore,
    output_path: Path,
    *,
    session_ids: Sequence[int] | None = None,
    limit: int | None = None,
    offset: int = 0,
    batch_size: int = DEFAULT_BATCH_SIZE,
    trajectory_limit: int | None = DEFAULT_TRAJECTORY_LIMIT,
    status_filters: Sequence[str] | None = None,
) -> ExportSummary:
    """Export sessions from storage into newline-delimited JSON."""

    stats = ExportSummary()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    statuses = {status.lower() for status in status_filters or []}
    with output_path.open("w", encoding="utf-8") as handle:
        async for payload in _iter_session_payloads(
            store,
            session_ids=session_ids,
            limit=limit,
            offset=offset,
            batch_size=batch_size,
            trajectory_limit=trajectory_limit,
            status_filters=statuses,
        ):
            if payload is None:
                continue
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            stats.sessions += 1
            stats.steps += len(payload.get("steps") or [])
    return stats


async def export_sessions_to_jsonl(
    database_url: str,
    output_path: str,
    *,
    session_ids: Sequence[int] | None = None,
    limit: int | None = None,
    offset: int = 0,
    batch_size: int = DEFAULT_BATCH_SIZE,
    trajectory_limit: int | None = DEFAULT_TRAJECTORY_LIMIT,
    status_filters: Sequence[str] | None = None,
    min_connections: int = 1,
    max_connections: int = 4,
    statement_timeout_seconds: float = 30.0,
) -> ExportSummary:
    """Connect to Postgres and export sessions to disk."""

    config = StorageConfig(
        database_url=database_url,
        min_connections=min_connections,
        max_connections=max_connections,
        statement_timeout_seconds=statement_timeout_seconds,
    )
    database = Database(config)
    await database.connect()
    try:
        stats = await export_sessions(
            database,
            Path(output_path),
            session_ids=session_ids,
            limit=limit,
            offset=offset,
            batch_size=batch_size,
            trajectory_limit=trajectory_limit,
            status_filters=status_filters,
        )
    finally:
        await database.disconnect()
    return stats


async def export_sessions_async(request: ExportRequest) -> ExportSummary:
    """Asynchronously export sessions using an ExportRequest."""

    return await export_sessions_to_jsonl(
        request.database_url,
        str(request.output_path),
        session_ids=request.session_ids,
        limit=request.limit,
        offset=request.offset,
        batch_size=request.batch_size,
        trajectory_limit=request.trajectory_event_limit,
        status_filters=request.status_filters,
        min_connections=request.min_connections,
        max_connections=request.max_connections,
        statement_timeout_seconds=request.statement_timeout_seconds,
    )


def export_sessions_sync(request: ExportRequest) -> ExportSummary:
    """Synchronous helper for CLI usage."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(export_sessions_async(request))
    raise RuntimeError("export_sessions_sync cannot run within an existing event loop")


async def _iter_session_payloads(
    store: SessionStore,
    *,
    session_ids: Sequence[int] | None,
    limit: int | None,
    offset: int,
    batch_size: int,
    trajectory_limit: int | None,
    status_filters: set[str],
):
    seen: set[int] = set()
    event_limit = (
        DEFAULT_TRAJECTORY_LIMIT if trajectory_limit is None or trajectory_limit <= 0 else trajectory_limit
    )

    if session_ids:
        for raw_id in session_ids:
            try:
                session_id = int(raw_id)
            except (TypeError, ValueError):
                logger.warning("Skipping invalid session id %s", raw_id)
                continue
            if session_id in seen:
                continue
            seen.add(session_id)
            payload = await _assemble_session(
                store,
                session_id,
                trajectory_limit=event_limit,
                status_filters=status_filters,
            )
            if payload is not None:
                yield payload
        return

    current_offset = max(offset, 0)
    remaining: Optional[int] = limit if limit is not None else None

    while remaining is None or remaining > 0:
        fetch_size = batch_size if remaining is None else min(batch_size, remaining)
        if fetch_size <= 0:
            break
        rows = await store.fetch_sessions(limit=fetch_size, offset=current_offset)
        if not rows:
            break
        for row in rows:
            session_id = row.get("id")
            if not isinstance(session_id, int):
                continue
            if session_id in seen:
                continue
            if status_filters:
                status_value = str(row.get("status", "")).lower()
                if status_value not in status_filters:
                    continue
            seen.add(session_id)
            payload = await _assemble_session(
                store,
                session_id,
                preloaded_session=row,
                trajectory_limit=event_limit,
                status_filters=status_filters,
            )
            if payload is not None:
                yield payload
        current_offset += len(rows)
        if remaining is not None:
            remaining -= len(rows)


async def _assemble_session(
    store: SessionStore,
    session_id: int,
    *,
    preloaded_session: dict[str, Any] | None = None,
    trajectory_limit: int | None = None,
    status_filters: set[str],
) -> dict[str, Any] | None:
    session_row = preloaded_session or await store.fetch_session(session_id)
    if session_row is None:
        logger.warning("Session %s not found; skipping", session_id)
        return None

    if status_filters:
        status_value = str(session_row.get("status", "")).lower()
        if status_value not in status_filters:
            return None

    detailed = session_row
    if "plan" not in detailed or detailed["plan"] is None:
        fetched = await store.fetch_session(session_id)
        if fetched is None:
            logger.warning("Session %s disappeared during export; skipping", session_id)
            return None
        detailed = fetched

    plan_model, plan_dict, plan_lookup = _coerce_plan(detailed.get("plan"))
    steps = await store.fetch_session_steps(session_id)
    events = await store.fetch_trajectory_events(
        session_id,
        limit=trajectory_limit if trajectory_limit is not None else DEFAULT_TRAJECTORY_LIMIT,
    )

    session_metadata = _coerce_dict(_coerce_json(detailed.get("metadata")))
    events_payload = _enrich_session_metadata(session_metadata, detailed, events)

    context_outputs: dict[int, Any] = {}
    step_payloads: list[dict[str, Any]] = []
    for step_row in steps:
        step_payload = _build_step_payload(step_row, plan_lookup, context_outputs)
        step_payloads.append(step_payload)
        output_value = step_payload.get("output") or ""
        metadata = step_payload.get("metadata") or {}
        step_id = step_payload.get("step_id")
        if isinstance(step_id, int):
            context_entry: dict[str, Any] = {"output_text": output_value}
            artifacts = metadata.get("artifacts")
            if artifacts is not None:
                context_entry["artifacts"] = artifacts
            status = metadata.get("status")
            if status is not None:
                context_entry["status"] = status
            notes = metadata.get("notes")
            if notes:
                context_entry["notes"] = notes
            context_outputs[step_id] = context_entry

    session_payload = {
        "task": detailed.get("task", ""),
        "final_answer": detailed.get("final_answer") or "",
        "plan": plan_dict if plan_dict else (plan_model.model_dump() if plan_model else {}),
        "steps": step_payloads,
        "session_metadata": session_metadata,
    }
    session_reward = _coerce_json(detailed.get("reward"))
    if isinstance(session_reward, dict) and session_reward:
        session_payload["session_reward"] = session_reward
    student_learning = detailed.get("student_learning") or session_metadata.get("student_learning")
    if isinstance(student_learning, str) and student_learning.strip():
        session_payload["student_learning"] = student_learning.strip()
    teacher_learning = detailed.get("teacher_learning") or session_metadata.get("teacher_learning")
    if isinstance(teacher_learning, str) and teacher_learning.strip():
        session_payload["teacher_learning"] = teacher_learning.strip()
    adaptive_summary = session_metadata.get("adaptive_summary")
    if isinstance(adaptive_summary, dict):
        session_payload["adaptive_summary"] = adaptive_summary
    triage_dossier = session_metadata.get("triage_dossier")
    if isinstance(triage_dossier, dict):
        session_payload["triage_dossier"] = triage_dossier
    personas_used = session_metadata.get("personas_used")
    if isinstance(personas_used, list):
        session_payload["personas_used"] = personas_used
    persona_updates = session_metadata.get("persona_updates")
    if persona_updates:
        session_payload["persona_updates"] = persona_updates
    persona_usage = session_metadata.get("persona_usage")
    if isinstance(persona_usage, dict) and persona_usage:
        session_payload["persona_usage"] = persona_usage
    teacher_notes = session_metadata.get("teacher_notes")
    if isinstance(teacher_notes, list):
        session_payload["teacher_notes"] = teacher_notes
    reward_summary = session_metadata.get("reward_summary")
    if isinstance(reward_summary, dict):
        session_payload["reward_summary"] = reward_summary
    if events_payload:
        session_payload["trajectory_events"] = events_payload
    return session_payload


def _enrich_session_metadata(
    session_metadata: dict[str, Any],
    session_row: dict[str, Any],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    session_id = session_row.get("id")
    status = session_row.get("status")
    created_at = session_row.get("created_at")
    completed_at = session_row.get("completed_at")

    if "session_id" not in session_metadata and session_id is not None:
        session_metadata["session_id"] = session_id
    if "status" not in session_metadata and status is not None:
        session_metadata["status"] = status
    if created_at is not None and not session_metadata.get("created_at"):
        session_metadata["created_at"] = _format_timestamp(created_at)
    if completed_at is not None and not session_metadata.get("completed_at"):
        session_metadata["completed_at"] = _format_timestamp(completed_at)

    normalised_events = [_normalise_event(event) for event in reversed(events)]
    if normalised_events:
        session_metadata["trajectory_events"] = normalised_events
    return normalised_events


def _build_step_payload(
    step_row: dict[str, Any],
    plan_lookup: dict[int, Step | dict[str, Any]],
    context_outputs: dict[int, Any],
) -> dict[str, Any]:
    step_id = step_row.get("step_id")
    plan_step = plan_lookup.get(step_id) if isinstance(step_id, int) else None

    raw_evaluation_value = step_row.get("evaluation")
    evaluation = _coerce_json(raw_evaluation_value) or {}
    reward_payload = evaluation.get("reward") or {}
    validation_payload = evaluation.get("validation") or {}

    guidance = step_row.get("guidance_notes") or []
    if not isinstance(guidance, list):
        guidance = [str(guidance)]

    attempt_details: list[dict[str, Any]] = []
    for attempt in step_row.get("attempt_details") or []:
        attempt_entry = dict(attempt)
        raw_eval = attempt_entry.get("evaluation")
        parsed_eval = _coerce_json(raw_eval)
        if parsed_eval is not None:
            attempt_entry["evaluation"] = parsed_eval
        attempt_details.append(attempt_entry)

    attempts_value = step_row.get("attempts")
    if isinstance(attempts_value, int):
        attempts = attempts_value
    else:
        try:
            attempts = int(attempts_value)
        except (TypeError, ValueError):
            attempts = len(attempt_details) or len(guidance) or 0

    metadata = _coerce_metadata(step_row.get("metadata"))
    if attempt_details:
        metadata.setdefault("attempt_history", attempt_details)
        metadata.setdefault("attempt_details", attempt_details)
    if evaluation:
        metadata.setdefault("raw_evaluation", evaluation)
    else:
        metadata.setdefault("raw_evaluation", {"raw": raw_evaluation_value})

    depends_on = _extract_plan_field(plan_step, "depends_on")
    if depends_on is not None and "depends_on" not in metadata:
        metadata["depends_on"] = list(depends_on)

    prior_results: dict[str, Any] = {}
    prior_results_text: dict[str, str] = {}
    for key, value in context_outputs.items():
        key_str = str(key)
        prior_results[key_str] = value
        if isinstance(value, dict):
            text_value = value.get("output_text")
            if text_value is None:
                text_value = value
        else:
            text_value = value
        if isinstance(text_value, (dict, list)):
            prior_results_text[key_str] = json.dumps(text_value, ensure_ascii=False)
        elif text_value is None:
            prior_results_text[key_str] = ""
        else:
            prior_results_text[key_str] = str(text_value)

    step_payload = {
        "step_id": step_id,
        "description": _extract_plan_field(plan_step, "description") or "",
        "trace": step_row.get("trace") or "",
        "output": step_row.get("output") or "",
        "reward": _build_reward_payload(reward_payload),
        "tool": _extract_plan_field(plan_step, "tool"),
        "tool_params": _coerce_dict(_extract_plan_field(plan_step, "tool_params")),
        "context": {"prior_results": prior_results, "prior_results_text": prior_results_text},
        "validation": _coerce_dict(validation_payload),
        "attempts": attempts,
        "guidance": [str(item) for item in guidance],
    }

    artifacts = metadata.get("artifacts")
    if artifacts is not None and "artifacts" not in step_payload:
        step_payload["artifacts"] = artifacts
    deliverable = metadata.get("deliverable")
    if deliverable is not None and "deliverable" not in step_payload:
        step_payload["deliverable"] = deliverable
    runtime_meta = metadata.get("runtime")
    if runtime_meta and "runtime" not in step_payload:
        step_payload["runtime"] = runtime_meta

    for key, value in metadata.items():
        if key not in step_payload:
            step_payload[key] = value

    step_payload["metadata"] = metadata
    return step_payload


def _build_reward_payload(entry: Any) -> dict[str, Any]:
    payload = _coerce_json(entry) or {}
    score = payload.get("score", 0.0)
    judges_data = payload.get("judges") or []
    judges: list[dict[str, Any]] = []
    for judge in judges_data:
        judge_payload = _coerce_json(judge) or {}
        samples_data = judge_payload.get("samples") or []
        samples: list[dict[str, Any]] = []
        for sample in samples_data:
            sample_payload = _coerce_json(sample) or {}
            samples.append(
                {
                    "score": _as_float(sample_payload.get("score")),
                    "rationale": sample_payload.get("rationale") or "",
                    "principles": sample_payload.get("principles") or [],
                    "uncertainty": _as_optional_float(sample_payload.get("uncertainty")),
                    "temperature": _as_optional_float(sample_payload.get("temperature")),
                }
            )
        judges.append(
            {
                "identifier": judge_payload.get("identifier") or "",
                "score": _as_float(judge_payload.get("score")),
                "rationale": judge_payload.get("rationale") or "",
                "principles": judge_payload.get("principles") or [],
                "samples": samples,
                "escalated": bool(judge_payload.get("escalated", False)),
                "escalation_reason": judge_payload.get("escalation_reason"),
            }
        )
    return {
        "score": _as_float(score),
        "judges": judges,
        "rationale": payload.get("rationale"),
        "raw": payload if isinstance(payload, dict) else {},
    }


def _coerce_plan(plan_data: Any) -> tuple[Plan | None, dict[str, Any], dict[int, Step | dict[str, Any]]]:
    if plan_data is None:
        return None, {}, {}
    parsed = _coerce_json(plan_data)
    if parsed is None:
        return None, {}, {}
    plan_lookup: dict[int, Step | dict[str, Any]] = {}
    try:
        plan_model = Plan.model_validate(parsed)
        plan_dict = plan_model.model_dump()
        plan_lookup = {step.id: step for step in plan_model.steps}
        return plan_model, plan_dict, plan_lookup
    except Exception:
        if isinstance(parsed, dict):
            steps = parsed.get("steps")
            if isinstance(steps, Iterable):
                for step in steps:
                    if isinstance(step, dict) and "id" in step:
                        try:
                            plan_lookup[int(step["id"])] = step
                        except (TypeError, ValueError):
                            continue
            return None, parsed if isinstance(parsed, dict) else {}, plan_lookup
    return None, {}, plan_lookup


def _extract_plan_field(plan_step: Step | dict[str, Any] | None, field: str) -> Any:
    if plan_step is None:
        return None
    if isinstance(plan_step, Step):
        return getattr(plan_step, field, None)
    if isinstance(plan_step, dict):
        return plan_step.get(field)
    return None


def _coerce_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return value


def _coerce_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    if isinstance(value, list):
        return {str(index): item for index, item in enumerate(value)}
    return {str(value): value}


def _coerce_metadata(value: Any) -> dict[str, Any]:
    parsed = _coerce_json(value)
    if isinstance(parsed, dict):
        return dict(parsed)
    if parsed is None:
        return {}
    return {"raw": parsed}


def _normalise_event(entry: dict[str, Any]) -> dict[str, Any]:
    payload = dict(entry)
    event_payload = _coerce_json(payload.get("event")) or {}
    payload["event"] = event_payload
    if isinstance(event_payload, dict):
        for key in ("type", "name"):
            if key in event_payload and key not in payload:
                payload[key] = event_payload[key]
    created_at = payload.get("created_at")
    if created_at is not None:
        payload["created_at"] = _format_timestamp(created_at)
    return payload


def _format_timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export Atlas runtime sessions to JSONL.")
    parser.add_argument("--database-url", required=True, help="PostgreSQL DSN for the Atlas session store.")
    parser.add_argument("--output", required=True, help="Path to the JSONL file to create.")
    parser.add_argument("--session-id", action="append", dest="session_ids", help="Limit export to specific session ids.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of sessions to export.")
    parser.add_argument("--offset", type=int, default=0, help="Number of sessions to skip before exporting.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for paginated session reads (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--trajectory-limit",
        type=int,
        default=DEFAULT_TRAJECTORY_LIMIT,
        help=f"Maximum trajectory events to include per session (default: {DEFAULT_TRAJECTORY_LIMIT}).",
    )
    parser.add_argument(
        "--status",
        action="append",
        dest="status_filters",
        help="Filter sessions by status (case-insensitive). May be provided multiple times.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress informational logs.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if not args.quiet else logging.WARNING, format="%(message)s")

    try:
        stats = asyncio.run(
            export_sessions_to_jsonl(
                args.database_url,
                args.output,
                session_ids=args.session_ids,
                limit=args.limit,
                offset=args.offset,
                batch_size=args.batch_size,
                trajectory_limit=args.trajectory_limit,
                status_filters=args.status_filters,
            )
        )
    except Exception as exc:  # pragma: no cover - CLI convenience
        logger.error("Failed to export sessions: %s", exc)
        return 1

    logger.info("Exported %s sessions (%s steps) to %s", stats.sessions, stats.steps, args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
