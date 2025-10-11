"""Helper CLI to inspect persona memory tables for the streaming SRE demo."""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import Any, Sequence

import asyncpg
from rich.console import Console
from rich.table import Table

DEFAULT_TENANT = "sre-demo"
console = Console()


async def _fetch_persona_rows(
    connection: asyncpg.Connection,
    tenant_id: str,
    persona_filter: Sequence[str] | None,
    limit: int,
) -> list[asyncpg.Record]:
    persona_clause = ""
    params: list[Any] = [tenant_id]
    if persona_filter:
        placeholders = ", ".join(f"${index}" for index in range(2, 2 + len(persona_filter)))
        persona_clause = f"AND pm.persona IN ({placeholders})"
        params.extend(persona_filter)
    params.append(limit)

    query = f"""
        SELECT
            pm.memory_id,
            pm.persona,
            pm.status,
            pm.trigger_fingerprint,
            pm.reward_snapshot,
            pm.retry_count,
            pm.created_at,
            COUNT(pmu.*) AS usage_count,
            AVG((pmu.reward->>'score')::float) AS avg_reward,
            AVG(pmu.retry_count::float) AS avg_retries
        FROM persona_memory pm
        LEFT JOIN persona_memory_usage pmu ON pm.memory_id = pmu.memory_id
        WHERE pm.tenant_id = $1
        {persona_clause}
        GROUP BY pm.memory_id
        ORDER BY pm.created_at DESC
        LIMIT ${len(params)}
    """
    return await connection.fetch(query, *params)


def _format_float(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):.2f}"


async def _render_report(database_url: str, tenant_id: str, persona_filter: Sequence[str], limit: int) -> None:
    conn = await asyncpg.connect(database_url)
    try:
        rows = await _fetch_persona_rows(conn, tenant_id, persona_filter, limit)
    finally:
        await conn.close()

    table = Table(title=f"Persona Memory Summary ({tenant_id})", expand=True)
    table.add_column("Memory ID", style="bold")
    table.add_column("Persona")
    table.add_column("Status")
    table.add_column("Usage Count", justify="right")
    table.add_column("Avg Reward", justify="right")
    table.add_column("Avg Retries", justify="right")
    table.add_column("Created At")

    if not rows:
        console.print("No persona memories found. Trigger a demo run first.")
        return

    for row in rows:
        table.add_row(
            str(row["memory_id"])[:8],
            row["persona"],
            row["status"],
            str(row["usage_count"]),
            _format_float(row["avg_reward"]),
            _format_float(row["avg_retries"]),
            row["created_at"].strftime("%Y-%m-%d %H:%M:%S"),
        )
    console.print(table)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect persona memory usage for the streaming SRE demo.")
    parser.add_argument(
        "--database-url",
        default=os.getenv("ATLAS_SRE_DEMO_DATABASE_URL"),
        help="Postgres DSN. Defaults to ATLAS_SRE_DEMO_DATABASE_URL.",
    )
    parser.add_argument(
        "--tenant-id",
        default=DEFAULT_TENANT,
        help="Tenant identifier to inspect.",
    )
    parser.add_argument(
        "--persona",
        action="append",
        default=None,
        help="Persona filter (can be specified multiple times).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of persona memories to display.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.database_url:
        console.print("[red]Database URL is required. Set ATLAS_SRE_DEMO_DATABASE_URL or pass --database-url.[/red]")
        return 1
    try:
        asyncio.run(_render_report(args.database_url, args.tenant_id, args.persona or None, args.limit))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

