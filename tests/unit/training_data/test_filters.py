"""Unit tests for training_data filters."""

from datetime import datetime

import pytest

from atlas.training_data.filters import build_session_query_filters


def test_build_filters_min_reward():
    """Test min_reward filter generation."""
    where_clause, params = build_session_query_filters(min_reward=0.8)

    assert "reward_stats IS NOT NULL" in where_clause
    assert "(reward_stats->>'score')::float >= $1" in where_clause
    assert params == [0.8]


def test_build_filters_created_after():
    """Test created_after filter generation."""
    dt = datetime(2024, 1, 1)
    where_clause, params = build_session_query_filters(created_after=dt)

    assert "created_at >= $1" in where_clause
    assert params == [dt]


def test_build_filters_learning_key():
    """Test learning_key filter generation."""
    where_clause, params = build_session_query_filters(learning_key="task-1")

    assert "(metadata->>'learning_key') = $1" in where_clause
    assert params == ["task-1"]


def test_build_filters_status():
    """Test status filter generation."""
    where_clause, params = build_session_query_filters(status_filters=["succeeded", "failed"])

    assert "status = ANY($1)" in where_clause
    assert params == [["succeeded", "failed"]]


def test_build_filters_review_status():
    """Test review_status filter generation."""
    where_clause, params = build_session_query_filters(review_status_filters=["approved"])

    assert "review_status = ANY($1)" in where_clause
    assert params == [["approved"]]


def test_build_filters_combined():
    """Test multiple filters combined."""
    dt = datetime(2024, 1, 1)
    where_clause, params = build_session_query_filters(
        min_reward=0.8,
        created_after=dt,
        learning_key="task-1",
        status_filters=["succeeded"],
    )

    assert "reward_stats IS NOT NULL" in where_clause
    assert "created_at >= $" in where_clause
    assert "(metadata->>'learning_key') = $" in where_clause
    assert "status = ANY($" in where_clause
    assert where_clause.count("AND") == 3
    assert len(params) == 4


def test_build_filters_empty():
    """Test empty filters return TRUE."""
    where_clause, params = build_session_query_filters()

    assert where_clause == "TRUE"
    assert params == []

