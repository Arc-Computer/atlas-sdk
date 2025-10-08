"""Command line entry point for the Atlas JSONL exporter."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from atlas.cli.jsonl_writer import ExportRequest, export_sessions_sync


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atlas.export",
        description="Export persisted Atlas runtime sessions to JSONL.",
    )
    parser.add_argument("--database-url", required=True, help="PostgreSQL connection URL.")
    parser.add_argument("--output", required=True, help="Destination JSONL file.")
    parser.add_argument(
        "--session-id",
        action="append",
        dest="session_ids",
        type=int,
        help="Specific session ID to export. Repeat for multiple sessions.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of sessions to export when no explicit IDs are provided (default: 50).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset applied when fetching recent sessions without explicit IDs.",
    )
    parser.add_argument(
        "--status",
        action="append",
        dest="statuses",
        help="Filter sessions by status. Repeat to allow multiple statuses.",
    )
    parser.add_argument(
        "--trajectory-event-limit",
        type=int,
        default=200,
        help="Maximum number of trajectory events to include per session.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational logs and only emit warnings/errors.",
    )
    return parser


def configure_logging(quiet: bool) -> None:
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.quiet)

    request = ExportRequest(
        database_url=args.database_url,
        output_path=Path(args.output).expanduser().resolve(),
        session_ids=args.session_ids,
        limit=args.limit,
        offset=args.offset,
        status_filters=args.statuses,
        trajectory_event_limit=args.trajectory_event_limit,
    )

    summary = export_sessions_sync(request)
    if summary.sessions == 0:
        logging.warning("No sessions were exported. Verify your filters and database contents.")
        return 1
    logging.info(
        "Completed export of %s sessions (%s steps).",
        summary.sessions,
        summary.steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
