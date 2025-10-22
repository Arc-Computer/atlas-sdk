"""Helpers for persisting autodiscovery telemetry into Postgres."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict

from atlas.config.models import StorageConfig
from atlas.runtime.storage.database import Database

logger = logging.getLogger(__name__)


def _database_url_from_env() -> str | None:
    """Return the configured database URL or None when persistence is disabled."""

    return os.getenv("STORAGE__DATABASE_URL") or os.getenv("DATABASE_URL")


async def _async_log_discovery_run(
    database_url: str,
    *,
    project_root: str,
    task: str,
    payload: Dict[str, Any],
    metadata: Dict[str, Any] | None,
    source: str,
) -> int:
    config = StorageConfig(
        database_url=database_url,
        min_connections=1,
        max_connections=1,
        statement_timeout_seconds=30.0,
    )
    database = Database(config)
    await database.connect()
    try:
        return await database.log_discovery_run(
            project_root=project_root,
            task=task,
            payload=payload,
            metadata=metadata,
            source=source,
        )
    finally:
        await database.disconnect()


def persist_discovery_run(
    *,
    task: str,
    project_root: Path | str,
    payload: Dict[str, Any],
    metadata: Dict[str, Any] | None = None,
    source: str,
) -> int | None:
    """Persist discovery telemetry if a database URL is configured.

    Returns the database identifier when persistence succeeds, otherwise None.
    Failures are logged but do not raise so the CLI remains best-effort.
    """

    database_url = _database_url_from_env()
    if not database_url:
        return None
    project_root_str = str(project_root)
    try:
        return asyncio.run(
            _async_log_discovery_run(
                database_url,
                project_root=project_root_str,
                task=task,
                payload=payload,
                metadata=metadata,
                source=source,
            )
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to persist discovery telemetry: %s", exc)
        return None


__all__ = ["persist_discovery_run"]
