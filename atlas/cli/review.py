"""Administrative helpers for the session quarantine and review workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

from atlas.config.models import StorageConfig
from atlas.runtime.storage.database import Database


def _coerce_dict(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {}
    return {}


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass(slots=True)
class ReviewSession:
    """Normalized session payload used for review CLI output."""

    identifier: int
    task: str
    status: str
    review_status: str
    review_notes: Optional[str]
    created_at: Optional[datetime]
    drift_alert: bool
    drift: Dict[str, Any]
    reward_stats: Dict[str, Any]
    metadata: Dict[str, Any]
    reward_audit: List[Dict[str, Any]]

    @property
    def score_delta(self) -> Optional[float]:
        return _coerce_float(self.drift.get("score_delta"))

    @property
    def uncertainty_delta(self) -> Optional[float]:
        return _coerce_float(self.drift.get("uncertainty_delta"))

    @property
    def score_z(self) -> Optional[float]:
        return _coerce_float(self.drift.get("score_z"))

    @property
    def uncertainty_z(self) -> Optional[float]:
        return _coerce_float(self.drift.get("uncertainty_z"))

    @property
    def score(self) -> Optional[float]:
        return _coerce_float(self.reward_stats.get("score") or self.reward_stats.get("score_mean"))

    @property
    def score_stddev(self) -> Optional[float]:
        return _coerce_float(self.reward_stats.get("score_stddev"))

    @property
    def sample_count(self) -> Optional[int]:
        return _coerce_int(self.reward_stats.get("sample_count"))


class ReviewClient:
    """Thin wrapper around the storage layer for review workflows."""

    def __init__(
        self,
        config: StorageConfig,
    ) -> None:
        self._database = Database(config)

    async def __aenter__(self) -> "ReviewClient":
        await self._database.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._database.disconnect()

    async def list_sessions(self, statuses: Sequence[str], *, limit: int, offset: int) -> Dict[str, List[ReviewSession]]:
        results: Dict[str, List[ReviewSession]] = {}
        for status in statuses:
            rows = await self._database.list_sessions_by_status(status, limit=limit, offset=offset)
            results[status] = [self._normalize_row(row) for row in rows]
        return results

    async def update_status(self, session_id: int, status: str, notes: str | None = None) -> None:
        await self._database.update_session_review_status(session_id, status, notes)

    async def fetch_session(self, session_id: int) -> Optional[ReviewSession]:
        row = await self._database.fetch_session(session_id)
        if row is None:
            return None
        return self._normalize_row(row)

    def _normalize_row(self, row: Dict[str, Any]) -> ReviewSession:
        metadata = _coerce_dict(row.get("metadata"))
        reward_stats_payload = row.get("reward_stats")
        reward_stats = _coerce_dict(reward_stats_payload) or _coerce_dict(metadata.get("reward_stats"))
        audit_payload = row.get("reward_audit") or metadata.get("reward_audit") or []
        reward_audit: List[Dict[str, Any]] = []
        if isinstance(audit_payload, list):
            for entry in audit_payload:
                if isinstance(entry, dict):
                    reward_audit.append(entry)
        drift_payload = metadata.get("drift") or {}
        drift = _coerce_dict(drift_payload)
        drift_alert = bool(
            metadata.get("drift_alert")
            or drift.get("drift_alert")
            or drift.get("alert")
        )
        identifier = _coerce_int(row.get("id")) or 0
        review_status = str(row.get("review_status") or metadata.get("review_status") or "pending")
        review_notes = row.get("review_notes")
        created_at = row.get("created_at")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at)
            except ValueError:
                created_at = None
        return ReviewSession(
            identifier=identifier,
            task=str(row.get("task") or metadata.get("task") or ""),
            status=str(row.get("status") or metadata.get("status") or ""),
            review_status=review_status,
            review_notes=str(review_notes).strip() if isinstance(review_notes, str) and review_notes.strip() else None,
            created_at=created_at if isinstance(created_at, datetime) else None,
            drift_alert=drift_alert,
            drift=drift,
            reward_stats=reward_stats,
            metadata=metadata,
            reward_audit=reward_audit,
        )


def format_review_groups(results: Dict[str, List[ReviewSession]]) -> List[str]:
    lines: List[str] = []
    for status, sessions in results.items():
        header = f"Status: {status} ({len(sessions)} session{'s' if len(sessions) != 1 else ''})"
        lines.append(header)
        if not sessions:
            lines.append("  (none)")
            continue
        for session in sessions:
            lines.extend(_format_session(session))
    return lines


def format_session_summary(session: ReviewSession) -> List[str]:
    return _format_session(session)


def _format_session(session: ReviewSession) -> List[str]:
    drift_state = "ALERT" if session.drift_alert else "ok"
    score_delta = _format_delta(session.score_delta)
    uncertainty_delta = _format_delta(session.uncertainty_delta)
    reward_display = _format_reward(session)
    created = session.created_at.isoformat() if session.created_at else "-"
    reason = session.drift.get("reason") or "-"
    first_line = (
        f"  {session.identifier:>6} | {drift_state:<5} | scoreΔ={score_delta} | uncΔ={uncertainty_delta} "
        f"| reward={reward_display} | created={created} | reason={reason}"
    )
    task_line = f"         Task: {session.task or '-'}"
    notes_line = f"         Notes: {session.review_notes or '-'}"
    audit_line = f"         Reward audit entries: {len(session.reward_audit)}"
    return [first_line, task_line, notes_line, audit_line]


def _format_delta(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2f}"


def _format_reward(session: ReviewSession) -> str:
    score = session.score
    stddev = session.score_stddev
    if score is None:
        reward = "n/a"
    elif stddev is not None:
        reward = f"{score:.2f}±{stddev:.2f}"
    else:
        reward = f"{score:.2f}"
    count = session.sample_count
    if count:
        reward = f"{reward} (n={count})"
    return reward


__all__ = [
    "ReviewClient",
    "ReviewSession",
    "format_review_groups",
    "format_session_summary",
]
