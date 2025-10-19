from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from atlas.config.models import StorageConfig
from atlas.runtime.learning_history import aggregate_learning_history
from atlas.runtime.storage.database import Database
from atlas.utils.env import load_dotenv_if_available


DEFAULT_OUTPUT = Path("probe_dataset.jsonl")


@dataclass(slots=True)
class SessionRecord:
    session_id: int
    task: str
    metadata: dict[str, Any]
    created_at: str | None
    status: str | None


def _parse_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return {}
    return {}


def _normalize_datetime(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, str):
        return value
    return None


def _extract_learning_key(metadata: dict[str, Any]) -> str | None:
    if not metadata:
        return None
    key = metadata.get("learning_key")
    if isinstance(key, str) and key:
        return key
    session_meta = metadata.get("session_metadata")
    if isinstance(session_meta, dict):
        key = session_meta.get("learning_key")
        if isinstance(key, str) and key:
            return key
    return None


def _expected_mode(metadata: dict[str, Any]) -> str | None:
    summary = metadata.get("adaptive_summary")
    if isinstance(summary, dict):
        mode = summary.get("adaptive_mode")
        if isinstance(mode, str):
            return mode
    # fall back to adaptive metadata if available
    adaptive = metadata.get("adaptive")
    if isinstance(adaptive, dict):
        mode = adaptive.get("active_mode")
        if isinstance(mode, str):
            return mode
    return None


async def _fetch_sessions(database: Database, limit: int, offset: int) -> list[SessionRecord]:
    rows = await database.fetch_sessions(limit=limit, offset=offset)
    records: list[SessionRecord] = []
    for row in rows:
        metadata = _parse_json(row.get("metadata"))
        records.append(
            SessionRecord(
                session_id=int(row["id"]),
                task=str(row.get("task") or ""),
                metadata=metadata,
                created_at=_normalize_datetime(row.get("created_at")),
                status=str(row.get("status")) if row.get("status") else None,
            )
        )
    return records


async def export_probe_dataset(
    *,
    database_url: str,
    output_path: Path,
    session_limit: int,
    session_offset: int,
    history_limit: int | None,
    min_history: int,
    include_missing_mode: bool,
    per_learning_key_limit: int | None,
) -> int:
    config = StorageConfig(database_url=database_url)
    database = Database(config)
    await database.connect()
    try:
        sessions = await _fetch_sessions(database, limit=session_limit, offset=session_offset)
        if not sessions:
            print("No sessions found for the specified window.")
            return 0

        output_path.parent.mkdir(parents=True, exist_ok=True)
        per_key_counts: dict[str, int] = {}
        written = 0
        with output_path.open("w", encoding="utf-8") as handle:
            for record in sessions:
                learning_key = _extract_learning_key(record.metadata)
                if not learning_key:
                    continue
                if per_learning_key_limit is not None:
                    count = per_key_counts.get(learning_key, 0)
                    if count >= per_learning_key_limit:
                        continue
                history_records = await database.fetch_learning_history(learning_key)
                if not history_records:
                    continue
                aggregated = aggregate_learning_history(history_records, limit=history_limit)
                if aggregated.get("total_count", 0) < min_history:
                    continue
                expected_mode = _expected_mode(record.metadata)
                if expected_mode is None and not include_missing_mode:
                    continue
                payload = {
                    "task": record.task,
                    "learning_history": aggregated,
                    "expected_mode": expected_mode,
                    "metadata": {
                        "session_id": record.session_id,
                        "created_at": record.created_at,
                        "status": record.status,
                    },
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                written += 1
                if per_learning_key_limit is not None:
                    per_key_counts[learning_key] = per_key_counts.get(learning_key, 0) + 1
    finally:
        await database.disconnect()
    return written


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export capability probe evaluation dataset from Postgres sessions.")
    parser.add_argument("--database-url", default=os.getenv("STORAGE__DATABASE_URL") or os.getenv("DATABASE_URL"), help="PostgreSQL URL (defaults to STORAGE__DATABASE_URL or DATABASE_URL).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"Destination JSONL file (default: {DEFAULT_OUTPUT}).")
    parser.add_argument("--limit", type=int, default=100, help="Number of sessions to inspect (default: 100).")
    parser.add_argument("--offset", type=int, default=0, help="Offset when fetching sessions (default: 0).")
    parser.add_argument("--history-limit", type=int, default=None, help="Override learning history window (default: use config/environment).")
    parser.add_argument("--min-history", type=int, default=1, help="Require at least this many historical entries (default: 1).")
    parser.add_argument("--include-missing-mode", action="store_true", help="Include sessions even when expected mode is missing.")
    parser.add_argument("--per-learning-key-limit", type=int, default=None, help="Maximum samples to include per learning key (default: unlimited).")
    return parser.parse_args(argv)


async def main_async(argv: Sequence[str] | None = None) -> int:
    load_dotenv_if_available()
    args = parse_args(argv)
    if not args.database_url:
        print("Error: database URL is required (set --database-url or STORAGE__DATABASE_URL).", file=sys.stderr)
        return 1
    written = await export_probe_dataset(
        database_url=args.database_url,
        output_path=args.output,
        session_limit=max(args.limit, 1),
        session_offset=max(args.offset, 0),
        history_limit=args.history_limit,
        min_history=max(args.min_history, 1),
        include_missing_mode=args.include_missing_mode,
        per_learning_key_limit=max(args.per_learning_key_limit, 1) if args.per_learning_key_limit else None,
    )
    print(f"Wrote {written} records to {args.output}")
    return 0 if written else 1


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(main_async(argv))


if __name__ == "__main__":
    raise SystemExit(main())
