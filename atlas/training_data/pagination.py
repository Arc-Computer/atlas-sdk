"""Pagination utilities for large result sets."""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

from atlas.runtime.schema import AtlasSessionTrace
from atlas.runtime.storage.database import Database


async def paginate_sessions(
    database: Database,
    filters: Dict[str, Any],
    *,
    batch_size: int = 1000,
    limit: Optional[int] = None,
) -> AsyncIterator[List[AtlasSessionTrace]]:
    """
    Paginate through large result sets without loading everything into memory.

    Args:
        database: Database instance (must be connected)
        filters: Dictionary of filter parameters matching get_training_sessions signature
        batch_size: Number of sessions to fetch per batch
        limit: Maximum total sessions to return (None for all)

    Yields:
        Batches of AtlasSessionTrace objects
    """
    from atlas.training_data.client import get_training_sessions_async
    from atlas.training_data.converters import convert_session_dict_to_trace

    offset = 0
    total_fetched = 0

    while True:
        # Calculate batch size for this iteration
        current_batch_size = batch_size
        if limit is not None:
            remaining = limit - total_fetched
            if remaining <= 0:
                break
            current_batch_size = min(batch_size, remaining)

        # Query this batch
        batch_filters = filters.copy()
        batch_filters["limit"] = current_batch_size
        batch_filters["offset"] = offset

        # Use query_training_sessions for efficiency
        session_rows = await database.query_training_sessions(
            min_reward=batch_filters.get("min_reward"),
            created_after=batch_filters.get("created_after"),
            learning_key=batch_filters.get("learning_key"),
            status_filters=batch_filters.get("status_filters"),
            review_status_filters=batch_filters.get("review_status_filters"),
            limit=current_batch_size,
            offset=offset,
        )

        if not session_rows:
            break

        # Convert to traces
        include_trajectory_events = batch_filters.get("include_trajectory_events", True)
        include_learning_data = batch_filters.get("include_learning_data", True)

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

        yield traces

        offset += len(session_rows)
        total_fetched += len(session_rows)

        # Stop if we've fetched all requested sessions or if batch was smaller than requested
        if limit is not None and total_fetched >= limit:
            break
        if len(session_rows) < current_batch_size:
            break

