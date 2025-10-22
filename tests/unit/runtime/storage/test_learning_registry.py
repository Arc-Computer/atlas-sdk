from __future__ import annotations

import json

import pytest

from atlas.config.models import StorageConfig
from atlas.runtime.storage.database import Database


class _FakeConnection:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple]] = []
        self.fetchrow_result = None

    async def execute(self, query: str, *params):
        self.executed.append((query, params))
        return None

    async def fetchrow(self, query: str, *params):
        self.executed.append((query, params))
        return self.fetchrow_result


class _PoolAcquire:
    def __init__(self, connection: _FakeConnection) -> None:
        self._connection = connection

    async def __aenter__(self):
        return self._connection

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakePool:
    def __init__(self, connection: _FakeConnection) -> None:
        self._connection = connection

    def acquire(self):
        return _PoolAcquire(self._connection)


@pytest.mark.asyncio
async def test_upsert_learning_state_serializes_metadata():
    config = StorageConfig(database_url="postgresql://atlas/example")
    database = Database(config)
    connection = _FakeConnection()
    database._pool = _FakePool(connection)  # type: ignore[attr-defined]

    await database.upsert_learning_state(
        "tenant::agent",
        "updated student pamphlet",
        "updated teacher pamphlet",
        {"version": 3},
    )

    assert len(connection.executed) == 1
    query, params = connection.executed[0]
    assert "INSERT INTO learning_registry" in query
    assert params[0] == "tenant::agent"
    assert params[1] == "updated student pamphlet"
    assert params[2] == "updated teacher pamphlet"
    assert json.loads(params[3]) == {"version": 3}


@pytest.mark.asyncio
async def test_fetch_learning_state_deserializes_metadata():
    config = StorageConfig(database_url="postgresql://atlas/example")
    database = Database(config)
    connection = _FakeConnection()
    connection.fetchrow_result = {
        "student_learning": "student pamphlet",
        "teacher_learning": None,
        "metadata": json.dumps({"version": 4}),
        "updated_at": "2025-03-01T10:00:00Z",
    }
    database._pool = _FakePool(connection)  # type: ignore[attr-defined]

    result = await database.fetch_learning_state("tenant::agent")

    query, params = connection.executed[0]
    assert "SELECT student_learning" in query
    assert params[0] == "tenant::agent"
    assert result["student_learning"] == "student pamphlet"
    assert result["teacher_learning"] is None
    assert result["metadata"] == {"version": 4}
    assert result["updated_at"] == "2025-03-01T10:00:00Z"
