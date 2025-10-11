"""Driver for the streaming SRE continual-learning demo."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import asyncpg
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from atlas.cli.jsonl_writer import ExportRequest, export_sessions_sync
from atlas.core import run as atlas_run
from atlas.runtime.orchestration.execution_context import ExecutionContext
from atlas.runtime.persona_memory import get_cache

from examples.streaming_sre_demo.data_stream import iter_incident_plan, render_incident

DEFAULT_CONFIG_PATH = Path("configs/examples/sre_demo_config.yaml")
EXPORT_ROOT = Path("examples/streaming_sre_demo/exports")
CSV_PATH = EXPORT_ROOT / "demo_results.csv"
JSONL_DIR = EXPORT_ROOT / "jsonl"

console = Console()


@dataclass
class IncidentMetrics:
    incident_id: int
    incident_type: str
    execution_mode: str
    diagnosis: str
    attempts: int
    retries: int
    reward: float | None
    tokens: int | None
    token_delta_pct: float | None
    reward_delta: float | None
    memories: list[str]
    candidate_ids: list[str]
    promoted_ids: list[str]
    demoted_ids: list[str]
    session_id: int | None
    duration_seconds: float
    status: str


def _parse_log_line(raw: str) -> dict[str, Any]:
    try:
        record = json.loads(raw)
        if isinstance(record, dict):
            return record
    except json.JSONDecodeError:
        pass
    return {"message": raw}


def _log_highlights(logs: Sequence[str], limit: int = 2) -> list[dict[str, Any]]:
    highlights: list[dict[str, Any]] = []
    for raw in logs[:limit]:
        record = _parse_log_line(raw)
        highlights.append(
            {
                "timestamp": record.get("timestamp"),
                "service": record.get("service"),
                "status": record.get("status"),
                "message": record.get("message"),
            }
        )
    return highlights


def _compose_summary(incident_payload: dict[str, Any]) -> str:
    metadata = incident_payload.get("metadata", {})
    severity = metadata.get("severity", "unknown")
    service = metadata.get("service") or metadata.get("incident_type", "")
    incident_type = metadata.get("incident_type", incident_payload.get("incident_type", ""))
    logs = incident_payload.get("logs") or []
    metrics = incident_payload.get("metrics") or []
    first_log = _parse_log_line(logs[0]) if logs else {}
    message = first_log.get("message")
    lead_metric = metrics[0] if metrics else {}
    metric_msg = ""
    if lead_metric:
        metric_msg = f" Key metric {lead_metric.get('metric')} at {lead_metric.get('value')} (baseline {lead_metric.get('baseline')})."
    return (
        f"{severity.title()} {incident_type.replace('_', ' ')} impacting {service}. "
        f"Primary symptom: {message or 'see logs'}.{metric_msg}"
    )


def _build_triaged_payload(incident_payload: dict[str, Any]) -> dict[str, Any]:
    metadata = incident_payload.get("metadata", {})
    highlights = _log_highlights(incident_payload.get("logs", []), limit=2)
    summary = _compose_summary(incident_payload)
    return {
        "incident_id": incident_payload.get("incident_id"),
        "summary": summary,
        "overview": {
            "incident_type": metadata.get("incident_type"),
            "severity": metadata.get("severity"),
            "service": metadata.get("service"),
            "tenant_id": metadata.get("tenant_id"),
        },
        "signals": {
            "log_highlights": highlights,
            "metrics": incident_payload.get("metrics", []),
            "recent_changes": incident_payload.get("recent_changes", []),
            "related_incidents": incident_payload.get("related_incidents", []),
        },
        "runbook_context": incident_payload.get("runbook_hint"),
        "customer_impact": metadata.get("customer_tier"),
    }


def _load_env_files() -> None:
    demo_env = Path(__file__).with_name(".env")
    root_env = Path(".env")
    for env_path in (demo_env, root_env):
        if env_path.exists():
            load_dotenv(env_path, override=False)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Atlas streaming SRE continual-learning demo.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the Atlas config file for the demo.",
    )
    parser.add_argument(
        "--speed",
        choices=("slow", "fast", "replay"),
        default="slow",
        help="Playback profile controlling delay between incidents.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Override delay between incidents in seconds.",
    )
    parser.add_argument(
        "--max-incidents",
        type=int,
        default=None,
        help="Limit the number of incidents processed (useful for dry runs).",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Disable JSONL exports (useful when Postgres is unavailable).",
    )
    parser.add_argument(
        "--no-cache-clear",
        action="store_true",
        help="Do not clear persona memory cache between incidents (developers only).",
    )
    return parser


def _resolve_database_url(config_path: Path) -> str:
    env_value = os.getenv("ATLAS_SRE_DEMO_DATABASE_URL")
    if env_value:
        return env_value
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    try:
        return payload["storage"]["database_url"]
    except KeyError as exc:
        raise RuntimeError("Unable to determine database_url. Set ATLAS_SRE_DEMO_DATABASE_URL in your .env.") from exc


def _playback_interval(speed: str, override: float | None) -> float:
    base = {"slow": 2.0, "fast": 0.75, "replay": 0.15}.get(speed.lower(), 2.0)
    if override and override > 0:
        return override
    return base


def _infer_diagnosis(final_answer: str) -> str:
    try:
        parsed = json.loads(final_answer)
        if isinstance(parsed, dict):
            for key in ("root_cause", "diagnosis", "result"):
                value = parsed.get(key)
                if isinstance(value, str) and value:
                    return value
    except json.JSONDecodeError:
        pass
    return final_answer.strip().splitlines()[0][:80]


def _extract_reward(result) -> float | None:
    score = None
    for step in result.step_results:
        reward = getattr(step.evaluation, "reward", None)
        candidate = getattr(reward, "score", None) if reward is not None else None
        if isinstance(candidate, (int, float)):
            score = float(candidate)
    return score


def _extract_tokens(result) -> int | None:
    tokens = None
    for step in result.step_results:
        metadata = step.metadata or {}
        counts = metadata.get("token_counts")
        if isinstance(counts, dict):
            approx = counts.get("approx_total") or counts.get("accumulated")
            if isinstance(approx, (int, float)):
                approx_int = int(approx)
                tokens = approx_int if tokens is None else max(tokens, approx_int)
    return tokens


def _extract_applied_memories(metadata: dict[str, Any]) -> list[str]:
    applied = metadata.get("applied_persona_memories") or {}
    summary: list[str] = []
    for persona, entries in applied.items():
        if not isinstance(entries, list):
            entries = [entries]
        for entry in entries:
            if isinstance(entry, dict):
                memory_id = entry.get("memory_id", "unknown")
                status = entry.get("status", "active")
            else:
                memory_id = entry
                status = "active"
            summary.append(f"{persona}:{str(memory_id)[:8]} ({status})")
    return summary


def _persona_for_memory(metadata: dict[str, Any], memory_id: str) -> str | None:
    memories = metadata.get("persona_memories") or {}
    for persona, records in memories.items():
        for record in records or []:
            if str(record.get("memory_id")) == str(memory_id):
                status = record.get("status", "unknown")
                return f"{persona} [{status}]"
    return None


def _append_memory_events(
    events: deque[str],
    metadata: dict[str, Any],
    metrics: IncidentMetrics,
) -> None:
    for candidate in metrics.candidate_ids:
        persona = _persona_for_memory(metadata, candidate) or "unknown persona"
        events.appendleft(f"candidate {candidate[:8]} created for {persona}")
    for promoted in metrics.promoted_ids:
        persona = _persona_for_memory(metadata, promoted) or "unknown persona"
        events.appendleft(f"promotion {promoted[:8]} → active ({persona})")
    for demoted in metrics.demoted_ids:
        persona = _persona_for_memory(metadata, demoted) or "unknown persona"
        events.appendleft(f"demoted {demoted[:8]} ({persona})")


async def _fetch_session_id(database_url: str, incident_id: int) -> int | None:
    conn = await asyncpg.connect(database_url)
    try:
        row = await conn.fetchrow(
            "SELECT id FROM sessions WHERE metadata->>'incident_id' = $1 ORDER BY created_at DESC LIMIT 1",
            str(incident_id),
        )
        if row:
            return int(row["id"])
    finally:
        await conn.close()
    return None


def _compute_delta(current: int | None, baseline: int | None) -> float | None:
    if current is None or baseline is None or baseline == 0:
        return None
    return ((current - baseline) / baseline) * 100.0


def _compute_reward_delta(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None:
        return None
    return current - baseline


def _format_delta(value: float | None, suffix: str = "%") -> str:
    if value is None:
        return "—"
    return f"{value:+.0f}{suffix}"


def _format_reward(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}"


def _format_tokens(value: int | None) -> str:
    if value is None:
        return "—"
    return f"{value:,}"


def _build_table(rows: Sequence[IncidentMetrics]) -> Table:
    table = Table(title="Streaming SRE Continual Learning", expand=True)
    table.add_column("Incident", style="bold white")
    table.add_column("Type", style="cyan")
    table.add_column("Mode", style="magenta")
    table.add_column("Attempts", justify="center")
    table.add_column("Reward")
    table.add_column("ΔReward")
    table.add_column("Tokens")
    table.add_column("ΔTokens")
    table.add_column("Diagnosis", style="green")
    table.add_column("Memories", style="yellow")

    for metrics in rows:
        style = "green" if metrics.attempts == 1 and (metrics.reward or 0) >= 0.7 else "red" if metrics.retries else "white"
        table.add_row(
            f"#{metrics.incident_id}",
            metrics.incident_type,
            metrics.execution_mode,
            str(metrics.attempts),
            _format_reward(metrics.reward),
            _format_delta(metrics.reward_delta, suffix=""),
            _format_tokens(metrics.tokens),
            _format_delta(metrics.token_delta_pct),
            metrics.diagnosis[:36],
            ", ".join(metrics.memories) or "—",
            style=style,
        )
    return table


def _build_layout(rows: Sequence[IncidentMetrics], memory_events: deque[str]) -> Layout:
    layout = Layout()
    layout.split(Layout(name="table", ratio=3), Layout(name="events", ratio=1))
    layout["table"].update(_build_table(rows))
    if memory_events:
        body = "\n".join(list(memory_events)[:10])
    else:
        body = "Awaiting persona memory activity…"
    layout["events"].update(Panel(Text(body), title="Persona Memory Events", border_style="blue"))
    return layout


def _append_csv(metrics: IncidentMetrics) -> None:
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    header = [
        "incident_id",
        "incident_type",
        "execution_mode",
        "attempts",
        "reward",
        "tokens",
        "token_delta_pct",
        "reward_delta",
        "session_id",
        "duration_seconds",
        "memories",
    ]
    write_header = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(header)
        writer.writerow(
            [
                metrics.incident_id,
                metrics.incident_type,
                metrics.execution_mode,
                metrics.attempts,
                metrics.reward if metrics.reward is not None else "",
                metrics.tokens if metrics.tokens is not None else "",
                metrics.token_delta_pct if metrics.token_delta_pct is not None else "",
                metrics.reward_delta if metrics.reward_delta is not None else "",
                metrics.session_id or "",
                f"{metrics.duration_seconds:.2f}",
                ";".join(metrics.memories),
            ]
        )


def _export_jsonl(database_url: str, session_id: int, incident_id: int) -> None:
    JSONL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = JSONL_DIR / f"incident_{incident_id:04d}.jsonl"
    request = ExportRequest(
        database_url=database_url,
        output_path=output_path,
        session_ids=[session_id],
        limit=1,
    )
    export_sessions_sync(request)


def _build_session_metadata(incident_payload: dict[str, Any], triaged_payload: dict[str, Any]) -> dict[str, Any]:
    metadata = incident_payload.get("metadata", {})
    tags = metadata.get("tags") or []
    return {
        "incident_id": str(incident_payload["incident_id"]),
        "tenant_id": metadata.get("tenant_id"),
        "tags": tags,
        "incident_type": metadata.get("incident_type"),
        "severity": metadata.get("severity"),
        "summary": triaged_payload.get("summary"),
    }


def _extract_attempts(result) -> int:
    attempts = 0
    for step in result.step_results:
        attempts = max(attempts, int(step.attempts or 0))
    return attempts or 1


def _collect_metrics(
    incident_payload: dict[str, Any],
    result,
    metadata: dict[str, Any],
    *,
    baseline_map: dict[str, tuple[int | None, float | None]],
    duration_seconds: float,
) -> IncidentMetrics:
    incident_type = incident_payload["metadata"]["incident_type"]
    reward = _extract_reward(result)
    tokens = _extract_tokens(result)
    attempts = _extract_attempts(result)
    retries = max(attempts - 1, 0)
    execution_mode = metadata.get("execution_mode", "unknown")
    baseline = baseline_map.get(incident_type)
    token_delta_pct = _compute_delta(tokens, baseline[0]) if baseline else None
    reward_delta = _compute_reward_delta(reward, baseline[1]) if baseline else None
    applied_memories = _extract_applied_memories(metadata)
    promotion = metadata.get("persona_promotion_result") or {}
    metrics = IncidentMetrics(
        incident_id=incident_payload["incident_id"],
        incident_type=incident_type,
        execution_mode=execution_mode,
        diagnosis=_infer_diagnosis(result.final_answer),
        attempts=attempts,
        retries=retries,
        reward=reward,
        tokens=tokens,
        token_delta_pct=token_delta_pct,
        reward_delta=reward_delta,
        memories=applied_memories,
        candidate_ids=[str(x) for x in metadata.get("new_persona_candidates") or []],
        promoted_ids=[str(x) for x in promotion.get("promoted") or []],
        demoted_ids=[str(x) for x in promotion.get("demoted") or []],
        session_id=None,
        duration_seconds=duration_seconds,
        status="succeeded",
    )
    if baseline is None and incident_payload["metadata"].get("novel"):
        baseline_map[incident_type] = (tokens, reward)
    return metrics


def run_demo(args: argparse.Namespace) -> None:
    _load_env_files()

    config_path = args.config.resolve()
    database_url = _resolve_database_url(config_path)
    interval = _playback_interval(args.speed, args.interval)

    plan = list(iter_incident_plan())
    if args.max_incidents:
        plan = plan[: args.max_incidents]

    memory_events: deque[str] = deque(maxlen=20)
    table_rows: list[IncidentMetrics] = []
    baseline_map: dict[str, tuple[int | None, float | None]] = {}

    console.print(f"[bold blue]Using config:[/bold blue] {config_path}")
    console.print(f"[bold blue]Postgres DSN:[/bold blue] {database_url}")
    console.print(f"[bold blue]Playback interval:[/bold blue] {interval:.2f}s")
    console.print()

    with Live(_build_layout(table_rows, memory_events), console=console, refresh_per_second=4) as live:
        for definition in plan:
            incident_payload = render_incident(definition)
            triaged_payload = _build_triaged_payload(incident_payload)
            task_payload = json.dumps(triaged_payload, ensure_ascii=False)
            session_metadata = _build_session_metadata(incident_payload, triaged_payload)

            start = time.perf_counter()
            try:
                result = atlas_run(
                    task=task_payload,
                    config_path=str(config_path),
                    session_metadata=session_metadata,
                    stream_progress=False,
                )
            except Exception as exc:  # pragma: no cover - runtime failure surface
                console.print(f"[red]Atlas run failed for incident {definition.incident_id}: {exc}[/red]")
                break
            duration = time.perf_counter() - start

            context = ExecutionContext.get()
            metadata = dict(context.metadata)

            metrics = _collect_metrics(incident_payload, result, metadata, baseline_map=baseline_map, duration_seconds=duration)

            if not args.skip_export:
                try:
                    metrics.session_id = asyncio.run(_fetch_session_id(database_url, metrics.incident_id))
                    if metrics.session_id:
                        _export_jsonl(database_url, metrics.session_id, metrics.incident_id)
                except Exception as export_exc:  # pragma: no cover - export failure surface
                    console.print(f"[yellow]Export failed for incident {metrics.incident_id}: {export_exc}[/yellow]")

            _append_memory_events(memory_events, metadata, metrics)
            table_rows.append(metrics)
            _append_csv(metrics)

            live.update(_build_layout(table_rows, memory_events))

            if not args.no_cache_clear:
                get_cache().clear()
            time.sleep(interval)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    try:
        run_demo(args)
    except KeyboardInterrupt:
        console.print("\n[cyan]Demo interrupted by user.[/cyan]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
