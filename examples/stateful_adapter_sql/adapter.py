"""Example stateful adapter that keeps a SQLite connection per session."""

from __future__ import annotations

import sqlite3
from typing import Any

from atlas.connectors.registry import SessionContext


class SQLiteAgent:
    """Simple stateful adapter that reuses an in-memory SQLite database."""

    def __init__(self) -> None:
        self._connection: sqlite3.Connection | None = None
        self._session_id: str | None = None

    async def on_open(self, context: SessionContext) -> dict[str, Any]:
        self._connection = sqlite3.connect(":memory:")
        self._session_id = context.session_id
        cursor = self._connection.cursor()
        cursor.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, title TEXT)")
        cursor.executemany(
            "INSERT INTO documents(title) VALUES (?)",
            [("Atlas SDK",), ("Continual Learning",), ("Stateful Sessions",)],
        )
        self._connection.commit()
        return {
            "database": "sqlite",
            "rows": 3,
            "session_id": self._session_id,
        }

    async def on_step(self, prompt: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._connection is None:
            raise RuntimeError("SQLiteAgent session has not been opened")
        cursor = self._connection.cursor()
        cursor.execute(prompt)
        rows = cursor.fetchall()
        return {
            "content": rows,
            "usage": {"prompt_tokens": 1, "completion_tokens": len(rows), "total_tokens": len(rows) + 1},
            "events": [{"type": "sql", "query": prompt}],
        }

    async def on_close(self, reason: str | None = None) -> dict[str, Any]:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        return {"closed": True, "reason": reason, "session_id": self._session_id}


__all__ = ["SQLiteAgent"]
