"""Command line entry point for the Atlas JSONL exporter and review tools."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from atlas.cli.jsonl_writer import DEFAULT_BATCH_SIZE, ExportRequest, export_sessions_sync
from atlas.cli.review import ReviewClient, ReviewSession, format_review_groups, format_session_summary
from atlas.utils.env import load_dotenv_if_available


def add_export_arguments(
    parser: argparse.ArgumentParser,
    *,
    require_database_url: bool = True,
    database_url_default: str | None = None,
    database_help: str | None = None,
    require_output: bool = True,
    output_default: str | None = None,
    output_help: str | None = None,
) -> argparse.ArgumentParser:
    """Attach standard export arguments to the provided parser."""

    parser.add_argument(
        "--database-url",
        required=require_database_url,
        default=database_url_default,
        help=database_help or "PostgreSQL connection URL.",
    )
    parser.add_argument(
        "--output",
        required=require_output,
        default=output_default,
        help=output_help or "Destination JSONL file.",
    )
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
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of sessions fetched per database query when paging results.",
    )
    parser.add_argument(
        "--include-status",
        action="append",
        dest="include_review_statuses",
        help="Include sessions with the given review status (default: approved only). Repeat to allow multiple statuses.",
    )
    parser.add_argument(
        "--include-all-statuses",
        action="store_true",
        help="Include sessions from every review status (overrides --include-status).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational logs and only emit warnings/errors.",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="atlas.export",
        description="Export persisted Atlas runtime sessions to JSONL.",
    )
    return add_export_arguments(parser)


def build_review_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="arc-atlas review",
        description="Inspect and manage session review status.",
    )
    subparsers = parser.add_subparsers(dest="review_command", required=True)

    sessions_parser = subparsers.add_parser("sessions", help="List sessions grouped by review status.")
    _add_database_url_argument(sessions_parser)
    sessions_parser.add_argument(
        "--status",
        action="append",
        dest="statuses",
        help="Specific review status to include (default: pending, quarantined, approved). Repeat to allow multiple statuses.",
    )
    sessions_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum sessions per status to display (default: 20).",
    )
    sessions_parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Offset applied when listing sessions for each status.",
    )
    sessions_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational logs.",
    )

    approve_parser = subparsers.add_parser("approve", help="Mark a session as approved.")
    _add_database_url_argument(approve_parser)
    approve_parser.add_argument("session_id", type=int, help="Session ID to approve.")
    approve_parser.add_argument(
        "--note",
        help="Optional review note explaining the approval decision.",
    )
    approve_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational logs.",
    )

    quarantine_parser = subparsers.add_parser("quarantine", help="Quarantine a session for further review.")
    _add_database_url_argument(quarantine_parser)
    quarantine_parser.add_argument("session_id", type=int, help="Session ID to quarantine.")
    quarantine_parser.add_argument(
        "--note",
        help="Optional review note explaining the quarantine decision.",
    )
    quarantine_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational logs.",
    )

    return parser


def _add_database_url_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--database-url",
        required=True,
        help="PostgreSQL connection URL.",
    )


def configure_logging(quiet: bool) -> None:
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def _run_export(args: Sequence[str] | None) -> int:
    parser = build_parser()
    parsed = parser.parse_args(args)
    configure_logging(parsed.quiet)

    env_statuses = os.getenv("ATLAS_REVIEW_DEFAULT_EXPORT_STATUSES")
    env_require = os.getenv("ATLAS_REVIEW_REQUIRE_APPROVAL")

    require_approval = True
    if env_require is not None:
        require_approval = env_require.strip().lower() not in {"0", "false", "off"}

    default_statuses: list[str] = ["approved"]
    include_all_default = False
    if env_statuses:
        tokens = [token.strip().lower() for token in env_statuses.split(",") if token.strip()]
        if any(token in {"*", "all"} for token in tokens):
            include_all_default = True
            default_statuses = []
        elif tokens:
            default_statuses = tokens
    elif not require_approval:
        # When approval gating is relaxed, include pending by default unless overridden.
        default_statuses = ["approved", "pending"]

    if parsed.include_all_statuses or include_all_default:
        review_status_filters = None
        include_all = True
    else:
        ordered_statuses: list[str] = []
        seen: set[str] = set()
        for status in [*default_statuses, *(parsed.include_review_statuses or [])]:
            normalized = str(status).lower()
            if normalized and normalized not in seen:
                seen.add(normalized)
                ordered_statuses.append(normalized)
        review_status_filters = ordered_statuses or None
        include_all = review_status_filters is None

    request = ExportRequest(
        database_url=parsed.database_url,
        output_path=Path(parsed.output).expanduser().resolve(),
        session_ids=parsed.session_ids,
        limit=parsed.limit,
        offset=parsed.offset,
        status_filters=parsed.statuses,
        review_status_filters=review_status_filters,
        include_all_review_statuses=include_all,
        trajectory_event_limit=parsed.trajectory_event_limit,
        batch_size=parsed.batch_size,
    )

    summary = export_sessions_sync(request)
    if summary.sessions == 0:
        logging.warning(
            "No sessions were exported. Approve sessions via 'arc-atlas review approve <id>' or, for local"
            " testing, set ATLAS_REVIEW_REQUIRE_APPROVAL=0 to bypass the approval gate."
        )
        return 1
    logging.info(
        "Completed export of %s sessions (%s steps).",
        summary.sessions,
        summary.steps,
    )
    return 0


def _run_review(args: Sequence[str]) -> int:
    parser = build_review_parser()
    parsed = parser.parse_args(args)
    configure_logging(getattr(parsed, "quiet", False))

    from atlas.config.models import StorageConfig  # Local import to avoid circular dependency at import time

    config = StorageConfig(
        database_url=parsed.database_url,
        min_connections=1,
        max_connections=2,
        statement_timeout_seconds=30.0,
    )

    if parsed.review_command == "sessions":
        statuses = [status.lower() for status in (parsed.statuses or ["pending", "quarantined", "approved"])]
        limit = max(parsed.limit, 1)
        offset = max(parsed.offset, 0)

        async def _list() -> Dict[str, List[ReviewSession]]:
            async with ReviewClient(config) as client:
                return await client.list_sessions(statuses, limit=limit, offset=offset)

        results = asyncio.run(_list())
        for line in format_review_groups(results):
            print(line)
        return 0

    if parsed.review_command in {"approve", "quarantine"}:
        target_status = "approved" if parsed.review_command == "approve" else "quarantined"
        session_id = parsed.session_id
        note = parsed.note

        async def _update() -> Optional[ReviewSession]:
            async with ReviewClient(config) as client:
                await client.update_status(session_id, target_status, note)
                return await client.fetch_session(session_id)

        summary = asyncio.run(_update())
        if summary is None:
            logging.error("Session %s was not found.", session_id)
            return 1
        print(f"Session {session_id} marked as {target_status}.")
        for line in format_session_summary(summary):
            print(line)
        return 0

    parser.error("Unknown review command")
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv_if_available()
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    if tokens and tokens[0] == "review":
        return _run_review(tokens[1:])
    if tokens and tokens[0] == "export":
        tokens = tokens[1:]
    return _run_export(tokens)


if __name__ == "__main__":
    raise SystemExit(main())
