"""Unit tests for training_data client functions."""

import json
from datetime import datetime
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atlas.runtime.schema import AtlasSessionTrace
from atlas.training_data.client import (
    count_training_sessions,
    get_session_by_id,
    get_training_sessions,
)


@pytest.fixture
def mock_database():
    """Create a mock database for testing."""
    db = MagicMock()
    db.connect = AsyncMock()
    db.disconnect = AsyncMock()
    db.close = AsyncMock()
    db.fetch_session = AsyncMock(return_value=None)
    db.fetch_session_steps = AsyncMock(return_value=[])
    db.fetch_trajectory_events = AsyncMock(return_value=[])
    db.query_training_sessions = AsyncMock(return_value=[])
    db._require_pool = MagicMock()
    db._require_pool.return_value.__aenter__ = AsyncMock()
    db._require_pool.return_value.__aexit__ = AsyncMock()
    return db


@pytest.fixture
def sample_session_dict():
    """Create a sample session dict."""
    return {
        "id": 1,
        "task": "test task",
        "final_answer": "answer",
        "plan": {"steps": []},
        "metadata": {},
        "reward": json.dumps({"score": 0.9}),
        "reward_stats": {"score": 0.9},
        "student_learning": "cue",
        "teacher_learning": "cue",
        "status": "succeeded",
        "review_status": "approved",
        "created_at": datetime.now(),
    }


@pytest.mark.asyncio
async def test_get_training_sessions_async_basic(mock_database, sample_session_dict):
    """Test basic get_training_sessions_async call."""
    from atlas.training_data.client import get_training_sessions_async

    mock_database.query_training_sessions.return_value = [sample_session_dict]

    with patch("atlas.training_data.client.Database", return_value=mock_database):
        with patch("atlas.training_data.client.StorageConfig"):
            sessions = await get_training_sessions_async(
                db_url="postgresql://test",
                limit=10,
            )

    assert isinstance(sessions, list)
    mock_database.connect.assert_called_once()
    mock_database.disconnect.assert_called_once()


@pytest.mark.asyncio
async def test_get_training_sessions_async_filters(mock_database, sample_session_dict):
    """Test get_training_sessions_async with filters."""
    from atlas.training_data.client import get_training_sessions_async

    mock_database.query_training_sessions.return_value = [sample_session_dict]

    with patch("atlas.training_data.client.Database", return_value=mock_database):
        with patch("atlas.training_data.client.StorageConfig"):
            await get_training_sessions_async(
                db_url="postgresql://test",
                min_reward=0.8,
                learning_key="task-1",
                status_filters=["succeeded"],
                limit=10,
            )

    mock_database.query_training_sessions.assert_called_once()
    call_kwargs = mock_database.query_training_sessions.call_args[1]
    assert call_kwargs["min_reward"] == 0.8
    assert call_kwargs["learning_key"] == "task-1"
    assert call_kwargs["status_filters"] == ["succeeded"]


@pytest.mark.asyncio
async def test_get_session_by_id_async(mock_database, sample_session_dict):
    """Test get_session_by_id_async."""
    from atlas.training_data.client import get_session_by_id_async

    mock_database.fetch_session.return_value = sample_session_dict

    with patch("atlas.training_data.client.Database", return_value=mock_database):
        with patch("atlas.training_data.client.StorageConfig"):
            session = await get_session_by_id_async(
                db_url="postgresql://test",
                session_id=1,
            )

    assert session is not None
    mock_database.fetch_session.assert_called_once_with(1)


@pytest.mark.asyncio
async def test_get_session_by_id_async_not_found(mock_database):
    """Test get_session_by_id_async when session not found."""
    from atlas.training_data.client import get_session_by_id_async

    mock_database.fetch_session.return_value = None

    with patch("atlas.training_data.client.Database", return_value=mock_database):
        with patch("atlas.training_data.client.StorageConfig"):
            session = await get_session_by_id_async(
                db_url="postgresql://test",
                session_id=999,
            )

    assert session is None


@pytest.mark.asyncio
async def test_count_training_sessions_async(mock_database):
    """Test count_training_sessions_async."""
    from atlas.training_data.client import count_training_sessions_async

    mock_pool = MagicMock()
    mock_connection = AsyncMock()
    mock_connection.fetchval = AsyncMock(return_value=42)
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_connection)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock()
    mock_database._require_pool.return_value = mock_pool

    with patch("atlas.training_data.client.Database", return_value=mock_database):
        with patch("atlas.training_data.client.StorageConfig"):
            count = await count_training_sessions_async(
                db_url="postgresql://test",
                min_reward=0.8,
            )

    assert count == 42


def test_get_training_sessions_sync_wrapper():
    """Test sync wrapper raises error in event loop."""
    with pytest.raises(RuntimeError, match="cannot run within an existing event loop"):
        # This will fail if run in an async context
        try:
            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(
                get_training_sessions(
                    db_url="postgresql://test",
                )
            )
        finally:
            loop.close()
            asyncio.set_event_loop(None)

