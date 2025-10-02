"""Storage access layer for the telemetry dashboard."""

from __future__ import annotations

from typing import Any

from atlas.storage.database import Database


class DashboardRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    async def connect(self) -> None:
        await self._database.connect()

    async def disconnect(self) -> None:
        await self._database.disconnect()

    async def list_sessions(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        return await self._database.fetch_sessions(limit=limit, offset=offset)

    async def get_session(self, session_id: int) -> dict[str, Any] | None:
        return await self._database.fetch_session(session_id)

    async def get_session_steps(self, session_id: int) -> list[dict[str, Any]]:
        return await self._database.fetch_session_steps(session_id)

    async def get_session_events(self, session_id: int, limit: int = 200) -> list[dict[str, Any]]:
        return await self._database.fetch_trajectory_events(session_id, limit)
