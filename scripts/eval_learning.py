from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Sequence

from atlas.config.models import StorageConfig
from atlas.evaluation.learning_report import (
    collect_learning_summaries,
    summary_to_dict,
    summary_to_markdown,
)
from atlas.runtime.storage.database import Database
from atlas.utils.env import load_dotenv_if_available


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate hint-less learning evaluation reports from persisted telemetry.",
    )
    parser.add_argument(
        "--database-url",
        required=True,
        help="PostgreSQL connection URL for the Atlas session store.",
    )
    parser.add_argument(
        "--learning-key",
        action="append",
        dest="learning_keys",
        help="Specific learning key to evaluate (repeatable). When omitted, the script selects recent keys automatically.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of learning keys to analyse when none are provided (default: 5).",
    )
    parser.add_argument(
        "--recent-window",
        type=int,
        default=5,
        help="Number of most recent sessions to include in the recent reward window (default: 5).",
    )
    parser.add_argument(
        "--baseline-window",
        type=int,
        default=50,
        help="Historical window size for baseline reward statistics (default: 50).",
    )
    parser.add_argument(
        "--discovery-limit",
        type=int,
        default=5,
        help="Maximum number of discovery/runtime run references to include per learning key (default: 5).",
    )
    parser.add_argument(
        "--trajectory-limit",
        type=int,
        default=200,
        help="Maximum number of trajectory events to inspect per session when counting telemetry (default: 200).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/learning"),
        help="Directory where summaries will be written (default: results/learning).",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Skip writing Markdown summaries (JSON outputs are always produced).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stdout progress messages.",
    )
    return parser.parse_args()


async def _load_learning_keys(database: Database, limit: int) -> list[str]:
    rows = await database.fetch_learning_keys(limit=limit)
    keys: list[str] = []
    for row in rows:
        key = row.get("learning_key")
        if isinstance(key, str) and key:
            keys.append(key)
    return keys


async def _gather_summaries(
    database: Database,
    learning_keys: Sequence[str],
    *,
    recent_window: int,
    baseline_window: int,
    discovery_limit: int,
    trajectory_limit: int,
):
    return await collect_learning_summaries(
        database,
        learning_keys,
        recent_window=recent_window,
        baseline_window=baseline_window,
        discovery_limit=discovery_limit,
        trajectory_limit=trajectory_limit,
    )


def _slug_for_key(learning_key: str) -> str:
    if not learning_key:
        return "learning_unknown"
    sanitized = "".join(ch for ch in learning_key if ch.isalnum())
    if sanitized:
        return sanitized[:16]
    digest = hashlib.sha256(learning_key.encode("utf-8")).hexdigest()
    return digest[:16]


def _write_outputs(
    summaries,
    output_dir: Path,
    *,
    write_markdown: bool,
) -> dict[str, dict[str, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, str]] = {}
    for summary in summaries:
        key = summary.learning_key
        slug = _slug_for_key(key)
        json_path = output_dir / f"{slug}_summary.json"
        json_payload = summary_to_dict(summary)
        json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
        record: dict[str, str] = {"json": str(json_path)}
        if write_markdown:
            markdown_path = output_dir / f"{slug}_summary.md"
            markdown_path.write_text(summary_to_markdown(summary), encoding="utf-8")
            record["markdown"] = str(markdown_path)
        manifest[key] = record
    index_path = output_dir / "index.json"
    index_payload = {
        "learning_keys": list(manifest.keys()),
        "artifacts": manifest,
    }
    index_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")
    return manifest


async def main() -> int:
    args = parse_args()
    load_dotenv_if_available()
    config = StorageConfig(
        database_url=args.database_url,
        min_connections=1,
        max_connections=4,
        statement_timeout_seconds=30.0,
    )
    database = Database(config)
    await database.connect()
    try:
        learning_keys = args.learning_keys or await _load_learning_keys(database, limit=max(args.limit, 1))
        if not learning_keys:
            if not args.quiet:
                print("No learning keys found; nothing to evaluate.")
            return 0
        summaries = await _gather_summaries(
            database,
            learning_keys,
            recent_window=max(args.recent_window, 1),
            baseline_window=max(args.baseline_window, 1),
            discovery_limit=max(args.discovery_limit, 1),
            trajectory_limit=max(args.trajectory_limit, 1),
        )
    finally:
        await database.disconnect()
    manifest = _write_outputs(
        summaries,
        args.output_dir,
        write_markdown=not args.no_markdown,
    )
    if not args.quiet:
        print(f"Generated learning summaries for {len(manifest)} learning keys in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
