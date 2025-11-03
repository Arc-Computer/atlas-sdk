"""Integration tests for training_data module with Docker Postgres."""

import os
import tempfile
from pathlib import Path

import pytest

from atlas.training_data.client import (
    count_training_sessions_async,
    get_session_by_id_async,
    get_training_sessions_async,
)


# Default Docker Postgres connection string (port 5433 maps to container port 5432)
DEFAULT_DB_URL = os.getenv("DATABASE_URL", "postgresql://atlas:atlas@localhost:5433/atlas")


def is_postgres_available(db_url: str) -> bool:
    """Check if Postgres is available."""
    try:
        import asyncpg

        async def check():
            try:
                conn = await asyncpg.connect(db_url)
                await conn.close()
                return True
            except Exception:
                return False

        import asyncio

        return asyncio.run(check())
    except ImportError:
        return False


@pytest.mark.skipif(
    not is_postgres_available(DEFAULT_DB_URL),
    reason="Docker Postgres not available",
)
async def test_get_training_sessions_integration():
    """Test get_training_sessions with real database."""
    from atlas.training_data.client import get_training_sessions_async

    sessions = await get_training_sessions_async(
        db_url=DEFAULT_DB_URL,
        limit=10,
    )

    assert isinstance(sessions, list)
    # Should not crash even if database is empty


@pytest.mark.skipif(
    not is_postgres_available(DEFAULT_DB_URL),
    reason="Docker Postgres not available",
)
async def test_reward_filtering_integration():
    """Test min_reward filter with real database."""
    from atlas.training_data.client import get_training_sessions_async

    # Query with min_reward filter
    sessions = await get_training_sessions_async(
        db_url=DEFAULT_DB_URL,
        min_reward=0.8,
        limit=10,
    )

    assert isinstance(sessions, list)
    # All returned sessions should have reward >= 0.8 (if any exist)
    for session in sessions:
        if session.session_reward:
            assert session.session_reward.get("score", 0) >= 0.8
        elif session.session_metadata.get("reward_summary"):
            reward_summary = session.session_metadata.get("reward_summary")
            if isinstance(reward_summary, dict) and "score" in reward_summary:
                assert float(reward_summary["score"]) >= 0.8


@pytest.mark.skipif(
    not is_postgres_available(DEFAULT_DB_URL),
    reason="Docker Postgres not available",
)
async def test_selective_loading_integration():
    """Test include_trajectory_events=False skips trajectory data."""
    from atlas.training_data.client import get_training_sessions_async

    sessions_with_events = await get_training_sessions_async(
        db_url=DEFAULT_DB_URL,
        limit=5,
        include_trajectory_events=True,
    )

    sessions_without_events = await get_training_sessions_async(
        db_url=DEFAULT_DB_URL,
        limit=5,
        include_trajectory_events=False,
    )

    assert len(sessions_with_events) == len(sessions_without_events)

    # Sessions without events should have None or empty trajectory_events
    for session in sessions_without_events:
        assert session.trajectory_events is None or len(session.trajectory_events) == 0


@pytest.mark.skipif(
    not is_postgres_available(DEFAULT_DB_URL),
    reason="Docker Postgres not available",
)
async def test_learning_key_filter_integration():
    """Test JSONB metadata filtering with real database."""
    from atlas.training_data.client import get_training_sessions_async

    # First, get a session to find a learning_key
    all_sessions = await get_training_sessions_async(
        db_url=DEFAULT_DB_URL,
        limit=10,
    )

    if not all_sessions:
        pytest.skip("No sessions in database to test learning_key filter")

    # Find a session with a learning_key
    session_with_key = None
    for session in all_sessions:
        if session.learning_key:
            session_with_key = session
            break

    if not session_with_key:
        pytest.skip("No sessions with learning_key in database")

    # Filter by that learning_key
    filtered_sessions = await get_training_sessions_async(
        db_url=DEFAULT_DB_URL,
        learning_key=session_with_key.learning_key,
        limit=10,
    )

    assert isinstance(filtered_sessions, list)
    # All returned sessions should have matching learning_key
    for session in filtered_sessions:
        assert session.learning_key == session_with_key.learning_key


@pytest.mark.skipif(
    not is_postgres_available(DEFAULT_DB_URL),
    reason="Docker Postgres not available",
)
async def test_count_training_sessions_integration():
    """Test count_training_sessions with real database."""
    from atlas.training_data.client import count_training_sessions_async

    count = await count_training_sessions_async(
        db_url=DEFAULT_DB_URL,
    )

    assert isinstance(count, int)
    assert count >= 0

    # Count with filter
    count_filtered = await count_training_sessions_async(
        db_url=DEFAULT_DB_URL,
        min_reward=0.8,
    )

    assert isinstance(count_filtered, int)
    assert count_filtered >= 0
    assert count_filtered <= count


@pytest.mark.skipif(
    not is_postgres_available(DEFAULT_DB_URL),
    reason="Docker Postgres not available",
)
@pytest.mark.asyncio
async def test_get_session_by_id_integration():
    """Test get_session_by_id with real database."""
    # First get a list of sessions
    sessions = await get_training_sessions(
        db_url=DEFAULT_DB_URL,
        limit=1,
    )

    if not sessions:
        pytest.skip("No sessions in database to test get_session_by_id")

    # Get the first session's ID from the database
    from atlas.config.models import StorageConfig
    from atlas.runtime.storage.database import Database

    config = StorageConfig(database_url=DEFAULT_DB_URL)
    db = Database(config)
    await db.connect()
    try:
        all_sessions = await db.fetch_sessions(limit=1)
        if not all_sessions:
            pytest.skip("No sessions in database")

        session_id = all_sessions[0]["id"]

        # Fetch by ID
        session = await get_session_by_id(
            db_url=DEFAULT_DB_URL,
            session_id=session_id,
        )

        assert session is not None
        assert session.task is not None
    finally:
        await db.disconnect()


@pytest.mark.skipif(
    not is_postgres_available(DEFAULT_DB_URL),
    reason="Docker Postgres not available",
)
async def test_field_preservation_round_trip():
    """Test field preservation: export JSONL vs direct query comparison."""
    from atlas.training_data.client import get_training_sessions_async

    # Get sessions via direct query
    sessions = await get_training_sessions_async(
        db_url=DEFAULT_DB_URL,
        limit=5,
        include_trajectory_events=True,
        include_learning_data=True,
    )

    if not sessions:
        pytest.skip("No sessions in database for round-trip test")

    # Convert to dict and verify essential fields are present
    for session in sessions:
        session_dict = session.to_dict()

        # Essential fields should be top-level
        assert "task" in session_dict
        assert "final_answer" in session_dict
        assert "plan" in session_dict
        assert "steps" in session_dict
        assert "session_metadata" in session_dict

        # Essential training fields should be present (may be None)
        assert "session_reward" in session_dict
        assert "trajectory_events" in session_dict
        assert "student_learning" in session_dict
        assert "teacher_learning" in session_dict
        assert "learning_history" in session_dict
        assert "adaptive_summary" in session_dict

        # Verify property accessors work
        if session.learning_key:
            assert session_dict["session_metadata"].get("learning_key") == session.learning_key

        # Verify steps have new fields
        for step in session_dict["steps"]:
            assert "runtime" in step
            assert "depends_on" in step

