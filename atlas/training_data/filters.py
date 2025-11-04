"""Build SQL WHERE clauses for session queries."""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional, Sequence, Tuple


def build_session_query_filters(
    min_reward: Optional[float] = None,
    created_after: Optional[datetime] = None,
    learning_key: Optional[str] = None,
    status_filters: Optional[Sequence[str]] = None,
    review_status_filters: Optional[Sequence[str]] = None,
) -> Tuple[str, List[Any]]:
    """
    Build WHERE clause and parameters for session queries.

    Returns: (WHERE clause, parameter list)

    SQL Pattern:
    - Reward filtering: (reward_stats->>'score')::float >= $1
    - Learning key: metadata->>'learning_key' = $2
    - Status: status = ANY($3)
    """
    constraints: List[str] = []
    params: List[Any] = []

    if min_reward is not None:
        params.append(min_reward)
        constraints.append(
            f"(reward_stats IS NOT NULL AND (reward_stats->>'score')::float >= ${len(params)})"
        )

    if created_after is not None:
        params.append(created_after)
        constraints.append(f"created_at >= ${len(params)}")

    if learning_key is not None:
        params.append(learning_key)
        constraints.append(f"(metadata->>'learning_key') = ${len(params)}")

    if status_filters:
        params.append(list(status_filters))
        constraints.append(f"status = ANY(${len(params)})")

    if review_status_filters:
        params.append(list(review_status_filters))
        constraints.append(f"review_status = ANY(${len(params)})")

    where_clause = " AND ".join(constraints) if constraints else "TRUE"
    return where_clause, params

