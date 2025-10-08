"""Export ARC demo sessions from Postgres to JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

from atlas.export.jsonl import ExportRequest, export_sessions_sync


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Atlas telemetry to JSONL.")
    parser.add_argument(
        "--database-url",
        required=True,
        help="PostgreSQL URL used by the demo run (e.g. postgresql://atlas:atlas@localhost:5432/atlas_arc_demo).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("arc_agi_traces.jsonl"),
        help="Path to the JSONL file that will be created.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional cap on the number of sessions exported.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    request = ExportRequest(
        database_url=args.database_url,
        output_path=args.output,
        limit=args.limit,
    )
    stats = export_sessions_sync(request)
    print(f"Exported {stats.sessions} sessions and {stats.steps} steps to {args.output}")


if __name__ == "__main__":
    main()
