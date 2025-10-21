"""Example stateful adapter that manages its own control loop and streams telemetry."""

from __future__ import annotations

import asyncio
import sqlite3
from typing import Any, Dict, Iterable, List, Sequence

from atlas.connectors import AdapterCapabilities, AdapterError, AdapterEventEmitter, AgentAdapter


class StatefulSQLiteAdapter(AgentAdapter):
    """Demonstration adapter that owns the inner loop and emits telemetry events."""

    def __init__(self) -> None:
        self._emit_event: AdapterEventEmitter | None = None
        self._db = sqlite3.connect(":memory:", check_same_thread=False)
        self._prepare_dataset()

    async def aopen_session(
        self,
        *,
        task: str,
        metadata: Dict[str, Any] | None = None,
        emit_event: AdapterEventEmitter | None = None,
    ) -> AdapterCapabilities:
        self._emit_event = emit_event
        await self._record_event(
            "progress",
            {"message": "Stateful adapter session initialised.", "task": task},
        )
        return AdapterCapabilities(control_loop="self", supports_stepwise=False, telemetry_stream=True)

    async def aplan(self, task: str, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
        await self._record_event("progress", {"message": "Adapter preparing single-shot plan."})
        description = (
            "Run the partner-managed task in a single exchange without Atlas prompt orchestration. "
            "Return the final answer after executing any required tool calls."
        )
        return {
            "steps": [
                {
                    "id": 1,
                    "description": description,
                    "tool": None,
                    "tool_params": None,
                    "depends_on": [],
                }
            ],
            "execution_mode": "single_shot",
        }

    async def aexecute(
        self,
        task: str,
        plan: Dict[str, Any],
        step: Dict[str, Any],
        metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        query = self._select_query(task)
        await self._record_event("env_action", {"sql": query, "reason": "Deriving answer from demo dataset."})
        rows = list(self._run_query(query))
        await self._record_event("tool_response", {"rows": rows})
        summary = self._format_rows(rows)
        await self._record_event("progress", {"message": "Adapter execution complete.", "rows": len(rows)})
        return {
            "trace": f"sqlite://memory?rows={len(rows)}",
            "output": summary,
            "metadata": {"rows": rows, "query": query, "status": "ok"},
            "deliverable": summary,
            "artifacts": {"table": rows},
            "status": "ok",
        }

    async def asynthesize(
        self,
        task: str,
        plan: Dict[str, Any],
        step_results: List[Dict[str, Any]],
        metadata: Dict[str, Any] | None = None,
    ) -> str:
        await self._record_event("progress", {"message": "Adapter synthesising final answer."})
        if step_results:
            deliverable = step_results[0].get("deliverable") or step_results[0].get("output") or ""
            return str(deliverable)
        return ""

    def _prepare_dataset(self) -> None:
        cursor = self._db.cursor()
        cursor.execute(
            "CREATE TABLE projects (name TEXT, owner TEXT, status TEXT, velocity INTEGER)"
        )
        cursor.executemany(
            "INSERT INTO projects VALUES (?, ?, ?, ?)",
            [
                ("Hermes", "Atlas SDK", "green", 42),
                ("Zephyr", "Atlas SDK", "yellow", 27),
                ("Orion", "Atlas SDK", "green", 33),
            ],
        )
        self._db.commit()

    def _select_query(self, task: str) -> str:
        lowered = task.lower()
        if "velocity" in lowered or "speed" in lowered:
            return "SELECT name, velocity FROM projects ORDER BY velocity DESC"
        if "count" in lowered or "how many" in lowered:
            return "SELECT status, COUNT(*) as total FROM projects GROUP BY status ORDER BY total DESC"
        return "SELECT name, owner, status FROM projects ORDER BY name"

    def _run_query(self, query: str) -> Iterable[tuple[Any, ...]]:
        cursor = self._db.cursor()
        cursor.execute(query)
        yield from cursor.fetchall()

    def _format_rows(self, rows: Sequence[Sequence[Any]]) -> str:
        if not rows:
            return "No matching records in the demo dataset."
        if len(rows[0]) == 4:
            return " | ".join(f"{name} (owner={owner}, status={status}, velocity={velocity})" for name, owner, status, velocity in rows)
        if len(rows[0]) == 3:
            return " | ".join(f"{name} (owner={owner}, status={status})" for name, owner, status in rows)
        if len(rows[0]) == 2:
            return " | ".join(f"{label}: {value}" for label, value in rows)
        return str(rows)

    async def _record_event(self, event: str, payload: Dict[str, Any] | None = None, *, reason: str | None = None) -> None:
        if self._emit_event is None:
            return
        try:
            await self._emit_event({"event": event, "payload": payload, "reason": reason})
        except Exception as exc:
            raise AdapterError(f"failed to emit adapter telemetry: {exc}") from exc


__all__ = ["StatefulSQLiteAdapter"]
