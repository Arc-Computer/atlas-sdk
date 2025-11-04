"""Unit tests for training_data pagination."""

from typing import AsyncIterator, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from atlas.runtime.schema import AtlasSessionTrace
from atlas.training_data.pagination import paginate_sessions


@pytest.fixture
def mock_database():
    """Create a mock database for testing."""
    db = MagicMock()
    db.fetch_session_steps = AsyncMock(return_value=[])
    db.fetch_trajectory_events = AsyncMock(return_value=[])
    return db


@pytest.mark.asyncio
async def test_paginate_sessions_basic(mock_database):
    """Test basic pagination."""
    # Mock two batches
    session_dicts_batch1 = [{"id": i, "task": f"task{i}", "plan": {"steps": []}, "metadata": {}} for i in range(1, 4)]
    session_dicts_batch2 = [{"id": i, "task": f"task{i}", "plan": {"steps": []}, "metadata": {}} for i in range(4, 6)]

    mock_database.query_training_sessions = AsyncMock(side_effect=[session_dicts_batch1, session_dicts_batch2, []])

    filters = {"include_trajectory_events": False, "include_learning_data": True}

    batches = []
    async for batch in paginate_sessions(mock_database, filters, batch_size=3):
        batches.append(batch)

    assert len(batches) == 2
    assert len(batches[0]) == 3
    assert len(batches[1]) == 2


@pytest.mark.asyncio
async def test_paginate_sessions_with_limit(mock_database):
    """Test pagination with limit."""
    session_dicts = [{"id": i, "task": f"task{i}", "plan": {"steps": []}, "metadata": {}} for i in range(1, 6)]

    mock_database.query_training_sessions = AsyncMock(side_effect=[session_dicts[:3], session_dicts[3:], []])

    filters = {"include_trajectory_events": False, "include_learning_data": True}

    batches = []
    async for batch in paginate_sessions(mock_database, filters, batch_size=3, limit=5):
        batches.append(batch)

    total = sum(len(b) for b in batches)
    assert total == 5


@pytest.mark.asyncio
async def test_paginate_sessions_empty_result(mock_database):
    """Test pagination with empty results."""
    mock_database.query_training_sessions = AsyncMock(return_value=[])

    filters = {"include_trajectory_events": False, "include_learning_data": True}

    batches = []
    async for batch in paginate_sessions(mock_database, filters, batch_size=10):
        batches.append(batch)

    assert len(batches) == 0

