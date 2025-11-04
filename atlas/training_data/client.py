"""Core query functions for training data access."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import List, Optional, Sequence

from atlas.config.models import StorageConfig
from atlas.runtime.schema import AtlasSessionTrace
from atlas.runtime.storage.database import Database

from .converters import convert_session_dict_to_trace
from .filters import build_session_query_filters


async def get_training_sessions_async(
    db_url: str,
    *,
    min_reward: float = 0.0,
    limit: int = 1000,
    offset: int = 0,
    created_after: Optional[datetime] = None,
    learning_key: Optional[str] = None,
    status_filters: Optional[Sequence[str]] = None,
    review_status_filters: Optional[Sequence[str]] = None,
    include_trajectory_events: bool = True,
    include_learning_data: bool = True,
    config: Optional[StorageConfig] = None,
) -> List[AtlasSessionTrace]:
    """
    Query training sessions directly from Postgres with selective data loading.

    Args:
        db_url: PostgreSQL connection string
        min_reward: Minimum reward score threshold (from reward_stats->>'score')
        limit: Maximum number of sessions to return
        offset: Pagination offset
        created_after: Filter sessions created after this timestamp
        learning_key: Filter by learning key (from metadata JSONB)
        status_filters: Filter by session status (e.g., ['succeeded', 'failed'])
        review_status_filters: Filter by review status (e.g., ['approved'])
        include_trajectory_events: Whether to fetch trajectory events (slower but complete)
        include_learning_data: Whether to include learning-related fields (reward, learning_history, etc.)
        config: Optional StorageConfig for connection pooling settings

    Returns:
        List of AtlasSessionTrace objects with all telemetry preserved
    """
    storage_config = config or StorageConfig(database_url=db_url)
    database = Database(storage_config)
    await database.connect()

    try:
        # Build filters
        where_clause, params = build_session_query_filters(
            min_reward=min_reward if min_reward != 0.0 else None,
            created_after=created_after,
            learning_key=learning_key,
            status_filters=status_filters,
            review_status_filters=review_status_filters,
        )

        # Query sessions with plan using the new method
        session_rows = await database.query_training_sessions(
            min_reward=min_reward if min_reward != 0.0 else None,
            created_after=created_after,
            learning_key=learning_key,
            status_filters=status_filters,
            review_status_filters=review_status_filters,
            limit=limit,
            offset=offset,
        )

        # For each session, fetch steps and trajectory events
        traces: List[AtlasSessionTrace] = []
        for session_row in session_rows:
            session_id = session_row.get("id")
            if not isinstance(session_id, int):
                continue

            # Fetch steps
            steps = await database.fetch_session_steps(session_id)

            # Fetch trajectory events if requested
            trajectory_events: List[dict[str, Any]] = []
            if include_trajectory_events:
                trajectory_events_raw = await database.fetch_trajectory_events(session_id)
                trajectory_events = [dict(row) for row in trajectory_events_raw]

            # Convert to trace
            trace = convert_session_dict_to_trace(
                session_row,
                steps,
                trajectory_events,
                include_learning_data=include_learning_data,
            )
            traces.append(trace)

        return traces

    finally:
        await database.disconnect()


def get_training_sessions(
    db_url: str,
    *,
    min_reward: float = 0.0,
    limit: int = 1000,
    offset: int = 0,
    created_after: Optional[datetime] = None,
    learning_key: Optional[str] = None,
    status_filters: Optional[Sequence[str]] = None,
    review_status_filters: Optional[Sequence[str]] = None,
    include_trajectory_events: bool = True,
    include_learning_data: bool = True,
    config: Optional[StorageConfig] = None,
) -> List[AtlasSessionTrace]:
    """
    Synchronous wrapper for get_training_sessions_async.

    See get_training_sessions_async for parameter documentation.
    """
    try:
        loop = asyncio.get_running_loop()
        raise RuntimeError(
            "get_training_sessions() cannot run within an existing event loop. "
            "Use get_training_sessions_async() instead."
        )
    except RuntimeError:
        # No running event loop, safe to create one
        return asyncio.run(
            get_training_sessions_async(
                db_url=db_url,
                min_reward=min_reward,
                limit=limit,
                offset=offset,
                created_after=created_after,
                learning_key=learning_key,
                status_filters=status_filters,
                review_status_filters=review_status_filters,
                include_trajectory_events=include_trajectory_events,
                include_learning_data=include_learning_data,
                config=config,
            )
        )


async def get_session_by_id_async(
    db_url: str,
    session_id: int,
    *,
    include_trajectory_events: bool = True,
    include_learning_data: bool = True,
    config: Optional[StorageConfig] = None,
) -> Optional[AtlasSessionTrace]:
    """
    Fetch single session by ID.

    Args:
        db_url: PostgreSQL connection string
        session_id: Session ID to fetch
        include_trajectory_events: Whether to fetch trajectory events
        include_learning_data: Whether to include learning-related fields
        config: Optional StorageConfig for connection pooling settings

    Returns:
        AtlasSessionTrace if found, None otherwise
    """
    storage_config = config or StorageConfig(database_url=db_url)
    database = Database(storage_config)
    await database.connect()

    try:
        session_row = await database.fetch_session(session_id)
        if session_row is None:
            return None

        # Fetch steps
        steps = await database.fetch_session_steps(session_id)

        # Fetch trajectory events if requested
        trajectory_events: List[dict[str, Any]] = []
        if include_trajectory_events:
            trajectory_events_raw = await database.fetch_trajectory_events(session_id)
            trajectory_events = [dict(row) for row in trajectory_events_raw]

        # Convert to trace
        trace = convert_session_dict_to_trace(
            session_row,
            steps,
            trajectory_events,
            include_learning_data=include_learning_data,
        )
        return trace

    finally:
        await database.disconnect()


def get_session_by_id(
    db_url: str,
    session_id: int,
    *,
    include_trajectory_events: bool = True,
    include_learning_data: bool = True,
    config: Optional[StorageConfig] = None,
) -> Optional[AtlasSessionTrace]:
    """
    Synchronous wrapper for get_session_by_id_async.

    See get_session_by_id_async for parameter documentation.
    """
    try:
        loop = asyncio.get_running_loop()
        raise RuntimeError(
            "get_session_by_id() cannot run within an existing event loop. "
            "Use get_session_by_id_async() instead."
        )
    except RuntimeError:
        return asyncio.run(
            get_session_by_id_async(
                db_url=db_url,
                session_id=session_id,
                include_trajectory_events=include_trajectory_events,
                include_learning_data=include_learning_data,
                config=config,
            )
        )


async def count_training_sessions_async(
    db_url: str,
    *,
    min_reward: float = 0.0,
    created_after: Optional[datetime] = None,
    learning_key: Optional[str] = None,
    status_filters: Optional[Sequence[str]] = None,
    review_status_filters: Optional[Sequence[str]] = None,
    config: Optional[StorageConfig] = None,
) -> int:
    """
    Count sessions matching criteria without loading data.

    Args:
        db_url: PostgreSQL connection string
        min_reward: Minimum reward score threshold
        created_after: Filter sessions created after this timestamp
        learning_key: Filter by learning key
        status_filters: Filter by session status
        review_status_filters: Filter by review status
        config: Optional StorageConfig for connection pooling settings

    Returns:
        Count of matching sessions
    """
    storage_config = config or StorageConfig(database_url=db_url)
    database = Database(storage_config)
    await database.connect()

    try:
        where_clause, params = build_session_query_filters(
            min_reward=min_reward if min_reward != 0.0 else None,
            created_after=created_after,
            learning_key=learning_key,
            status_filters=status_filters,
            review_status_filters=review_status_filters,
        )

        query = f"SELECT COUNT(*) FROM sessions WHERE {where_clause}"

        pool = database._require_pool()
        async with pool.acquire() as connection:
            count = await connection.fetchval(query, *params)
            return int(count) if count is not None else 0

    finally:
        await database.disconnect()


def count_training_sessions(
    db_url: str,
    *,
    min_reward: float = 0.0,
    created_after: Optional[datetime] = None,
    learning_key: Optional[str] = None,
    status_filters: Optional[Sequence[str]] = None,
    review_status_filters: Optional[Sequence[str]] = None,
    config: Optional[StorageConfig] = None,
) -> int:
    """
    Synchronous wrapper for count_training_sessions_async.

    See count_training_sessions_async for parameter documentation.
    """
    try:
        loop = asyncio.get_running_loop()
        raise RuntimeError(
            "count_training_sessions() cannot run within an existing event loop. "
            "Use count_training_sessions_async() instead."
        )
    except RuntimeError:
        return asyncio.run(
            count_training_sessions_async(
                db_url=db_url,
                min_reward=min_reward,
                created_after=created_after,
                learning_key=learning_key,
                status_filters=status_filters,
                review_status_filters=review_status_filters,
                config=config,
            )
        )

