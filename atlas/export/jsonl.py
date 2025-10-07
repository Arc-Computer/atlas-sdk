"""Utilities for exporting persisted runtime sessions as JSONL traces."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pydantic import ValidationError

from atlas.config.models import StorageConfig
from atlas.storage.database import Database
from atlas.types import Plan

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ExportRequest:
    """Parameters controlling the export process."""

    database_url: str
    output_path: Path
    session_ids: Sequence[int] | None = None
    limit: int | None = None
    offset: int = 0
    status_filters: Sequence[str] | None = None
    trajectory_event_limit: int = 200


@dataclass(slots=True)
class ExportSummary:
    sessions: int = 0
    steps: int = 0

    def merge(self, other: "ExportSummary") -> None:
        self.sessions += other.sessions
        self.steps += other.steps


async def export_sessions(request: ExportRequest) -> ExportSummary:
    """Export matching sessions to newline-delimited JSON."""

    database = Database(StorageConfig(database_url=request.database_url))
    await database.connect()
    summary = ExportSummary()
    try:
        session_ids = await _resolve_session_ids(database, request)
        if not session_ids:
            logger.info("No sessions matched the provided filters.")
            return summary

        output_path = request.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as handle:
            for session_id in session_ids:
                record = await _build_session_record(database, session_id, request.trajectory_event_limit)
                if record is None:
                    logger.warning("Skipping session %s because it could not be retrieved.", session_id)
                    continue
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")
                summary.sessions += 1
                summary.steps += len(record.get("steps") or [])
        logger.info(
            "Exported %s sessions (%s steps) to %s",
            summary.sessions,
            summary.steps,
            output_path,
        )
        return summary
    finally:
        await database.disconnect()


def export_sessions_sync(request: ExportRequest) -> ExportSummary:
    """Synchronous helper for CLI usage."""

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(export_sessions(request))
    raise RuntimeError("export_sessions_sync cannot run within an existing event loop")


async def _resolve_session_ids(database: Database, request: ExportRequest) -> List[int]:
    if request.session_ids:
        return list(dict.fromkeys(int(sid) for sid in request.session_ids))

    limit = request.limit or 50
    rows = await database.fetch_sessions(limit=limit, offset=request.offset)
    filtered: List[int] = []
    statuses = {status.lower() for status in request.status_filters or []}
    for row in rows:
        if statuses and str(row.get("status", "")).lower() not in statuses:
            continue
        filtered.append(int(row["id"]))
    return filtered


async def _build_session_record(
    database: Database,
    session_id: int,
    trajectory_event_limit: int,
) -> Optional[Dict[str, Any]]:
    session = await database.fetch_session(session_id)
    if session is None:
        return None

    plan_model = _parse_plan(session.get("plan"))
    plan_index = _index_plan(plan_model)

    steps_raw = await database.fetch_session_steps(session_id)
    events_raw = await database.fetch_trajectory_events(session_id, limit=trajectory_event_limit)

    session_metadata = _coerce_dict(session.get("metadata"))
    session_metadata = dict(session_metadata)
    session_metadata.setdefault("status", session.get("status"))
    session_metadata.setdefault("created_at", _iso_or_none(session.get("created_at")))
    session_metadata.setdefault("completed_at", _iso_or_none(session.get("completed_at")))

    steps = [_build_step_record(row, plan_index) for row in steps_raw]

    record: Dict[str, Any] = {
        "session_id": session_id,
        "task": session.get("task") or "",
        "final_answer": session.get("final_answer") or "",
        "plan": plan_model.model_dump() if plan_model else _plan_payload(session.get("plan")),
        "steps": steps,
        "session_metadata": session_metadata,
    }

    if events_raw:
        record["trajectory_events"] = [_coerce_dict(row.get("event")) for row in events_raw]

    return record


def _parse_plan(raw_plan: Any) -> Plan | None:
    if raw_plan is None:
        return None
    if isinstance(raw_plan, str):
        try:
            raw_plan = json.loads(raw_plan)
        except json.JSONDecodeError:
            return None
    try:
        return Plan.model_validate(raw_plan)
    except ValidationError:
        return None


def _plan_payload(raw_plan: Any) -> Dict[str, Any]:
    if raw_plan is None:
        return {}
    if isinstance(raw_plan, dict):
        return raw_plan
    if isinstance(raw_plan, str):
        try:
            parsed = json.loads(raw_plan)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _index_plan(plan: Plan | None) -> Dict[int, Dict[str, Any]]:
    if plan is None:
        return {}
    index: Dict[int, Dict[str, Any]] = {}
    for step in plan.steps:
        index[step.id] = {
            "description": step.description,
            "tool": step.tool,
            "tool_params": step.tool_params or {},
        }
    return index


def _build_step_record(row: Dict[str, Any], plan_index: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    step_id = row.get("step_id", 0)
    plan_entry = plan_index.get(step_id, {})
    evaluation = _coerce_dict(row.get("evaluation"))
    reward = _coerce_reward(evaluation.get("reward"))
    validation = _coerce_dict(evaluation.get("validation"))
    metadata = _coerce_dict(row.get("metadata"))
    if row.get("attempt_details"):
        metadata.setdefault("attempt_details", row["attempt_details"])

    return {
        "step_id": step_id,
        "description": plan_entry.get("description", ""),
        "trace": row.get("trace") or "",
        "output": row.get("output") or "",
        "reward": reward,
        "tool": plan_entry.get("tool"),
        "tool_params": plan_entry.get("tool_params") or {},
        "context": {},
        "validation": validation,
        "attempts": row.get("attempts") or 0,
        "guidance": row.get("guidance_notes") or [],
        "metadata": metadata,
    }


def _coerce_reward(raw_reward: Any) -> Dict[str, Any]:
    payload = _coerce_dict(raw_reward)
    judges: List[Dict[str, Any]] = []
    for entry in payload.get("judges", []) or []:
        entry_dict = entry if isinstance(entry, dict) else _coerce_dict(entry)
        samples: List[Dict[str, Any]] = []
        for sample in entry_dict.get("samples", []) or []:
            sample_dict = sample if isinstance(sample, dict) else _coerce_dict(sample)
            samples.append(
                {
                    "score": _coerce_float(sample_dict, "score"),
                    "rationale": _coerce_str(sample_dict, "rationale"),
                    "principles": sample_dict.get("principles") or [],
                    "uncertainty": sample_dict.get("uncertainty"),
                    "temperature": sample_dict.get("temperature"),
                }
            )
        judges.append(
            {
                "identifier": entry_dict.get("identifier", ""),
                "score": _coerce_float(entry_dict, "score"),
                "rationale": _coerce_str(entry_dict, "rationale"),
                "principles": entry_dict.get("principles") or [],
                "samples": samples,
                "escalated": bool(entry_dict.get("escalated", False)),
                "escalation_reason": entry_dict.get("escalation_reason"),
            }
        )
    return {
        "score": _coerce_float(payload, "score"),
        "rationale": payload.get("rationale"),
        "judges": judges,
        "raw": payload if payload else {},
    }


def _coerce_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _coerce_str(source: Any, key: str) -> str:
    if isinstance(source, dict):
        value = source.get(key)
    else:
        value = None
    return str(value) if value is not None else ""


def _coerce_float(source: Any, key: str) -> float:
    if isinstance(source, dict):
        value = source.get(key)
    else:
        value = source
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _iso_or_none(value: Any) -> Optional[str]:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        return value
    return None
