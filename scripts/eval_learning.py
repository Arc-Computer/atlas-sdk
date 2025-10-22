from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

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
        "--batch-size",
        type=int,
        default=4,
        help="Maximum number of learning keys to evaluate concurrently (default: 4).",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip per-session trajectory fetches and generate lightweight summaries.",
    )
    parser.add_argument(
        "--filter-project",
        help="Restrict evaluation to sessions captured under the given project root.",
    )
    parser.add_argument(
        "--filter-task",
        help="Restrict evaluation to a single task name.",
    )
    parser.add_argument(
        "--filter-tag",
        action="append",
        dest="filter_tags",
        default=None,
        help="Restrict evaluation to sessions containing the given tag (repeatable).",
    )
    parser.add_argument(
        "--compare-to",
        type=Path,
        help="Path to a previous results/learning/index.json to compute deltas against.",
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


async def _load_learning_keys(
    database: Database,
    *,
    limit: int | None,
    project_root: str | None,
    task: str | None,
    tags: Sequence[str] | None,
) -> list[str]:
    rows = await database.fetch_learning_keys(
        limit=limit,
        project_root=project_root,
        task=task,
        tags=tags,
    )
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
    summary_only: bool,
    project_root: str | None,
    task_filter: str | None,
    tags: Sequence[str] | None,
    max_concurrency: int,
    session_limit: int | None = None,
):
    return await collect_learning_summaries(
        database,
        learning_keys,
        recent_window=recent_window,
        baseline_window=baseline_window,
        discovery_limit=discovery_limit,
        trajectory_limit=trajectory_limit,
        summary_only=summary_only,
        project_root=project_root,
        task_filter=task_filter,
        tags=tags,
        max_concurrency=max_concurrency,
        session_limit=session_limit,
    )


def _slug_for_key(learning_key: str) -> str:
    if not learning_key:
        return "learning_unknown"
    sanitized = "".join(ch for ch in learning_key if ch.isalnum())
    if sanitized:
        return sanitized[:16]
    digest = hashlib.sha256(learning_key.encode("utf-8")).hexdigest()
    return digest[:16]


def _load_previous_summaries(index_path: Path) -> dict[str, dict[str, Any]]:
    if not index_path.exists():
        raise FileNotFoundError(f"Comparison index not found: {index_path}")
    base_dir = index_path.parent
    manifest = json.loads(index_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", {})
    summaries: dict[str, dict[str, Any]] = {}
    for learning_key, entry in artifacts.items():
        json_path_str = entry.get("json")
        if not isinstance(json_path_str, str):
            continue
        json_path = Path(json_path_str)
        candidate_paths = [json_path]
        if not json_path.is_absolute():
            candidate_paths.append(base_dir / json_path)
        resolved_path: Path | None = None
        for candidate in candidate_paths:
            if candidate.exists():
                resolved_path = candidate
                break
            try:
                absolute_candidate = candidate.resolve()
            except FileNotFoundError:
                continue
            if absolute_candidate.exists():
                resolved_path = absolute_candidate
                break
        if resolved_path is None:
            continue
        try:
            payload = json.loads(resolved_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        summaries[learning_key] = payload
    return summaries


def _compute_delta(current: float | int | None, previous: float | int | None) -> float | int | None:
    if current is None or previous is None:
        return None
    return current - previous


def _compute_comparisons(
    summaries: Sequence[Any],
    previous_payloads: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    comparisons: dict[str, dict[str, Any]] = {}
    recent_deltas: list[dict[str, Any]] = []
    session_deltas: list[dict[str, Any]] = []

    for summary in summaries:
        previous = previous_payloads.get(summary.learning_key)
        if not previous:
            continue
        previous_reward = previous.get("reward", {})
        previous_models = {
            (entry.get("role"), entry.get("model_id")): entry
            for entry in previous.get("model_breakdown", [])
            if isinstance(entry, dict)
        }
        recent_mean_delta = _compute_delta(summary.reward.recent_mean, previous_reward.get("recent_mean"))
        latest_score_delta = _compute_delta(summary.reward.latest_score, previous_reward.get("latest_score"))
        session_count_delta = summary.session_count - int(previous.get("session_count", 0))

        model_deltas: dict[str, dict[str, Any]] = {}
        for entry in summary.model_breakdown:
            key = (entry.role, entry.model_id)
            prev_entry = previous_models.pop(key, None)
            prev_session_count = int(prev_entry.get("session_count", 0)) if prev_entry else 0
            prev_reward_mean = prev_entry.get("reward_mean") if prev_entry else None
            delta_sessions = entry.session_count - prev_session_count
            delta_reward_mean = _compute_delta(entry.reward_mean, prev_reward_mean)
            if delta_sessions or delta_reward_mean is not None:
                model_deltas[f"{entry.role}:{entry.model_id}"] = {
                    "session_count_delta": delta_sessions,
                    "reward_mean_delta": delta_reward_mean,
                }
        for key, prev_entry in previous_models.items():
            role, model_id = key
            prev_session_count = int(prev_entry.get("session_count", 0))
            prev_reward_mean = prev_entry.get("reward_mean")
            model_deltas[f"{role}:{model_id}"] = {
                "session_count_delta": -prev_session_count,
                "reward_mean_delta": _compute_delta(None, prev_reward_mean),
            }

        comparison_entry = {
            "recent_mean_delta": recent_mean_delta,
            "latest_score_delta": latest_score_delta,
            "session_count_delta": session_count_delta,
            "model_deltas": model_deltas,
        }
        comparisons[summary.learning_key] = comparison_entry

        if recent_mean_delta is not None:
            recent_deltas.append({"learning_key": summary.learning_key, "delta": recent_mean_delta})
        session_deltas.append({"learning_key": summary.learning_key, "delta": session_count_delta})

    recent_sorted = sorted(recent_deltas, key=lambda item: item["delta"])
    session_sorted = sorted(session_deltas, key=lambda item: item["delta"])
    aggregate = {
        "recent_mean_best": list(reversed(recent_sorted[-5:])),
        "recent_mean_worst": recent_sorted[:5],
        "session_growth": list(reversed([entry for entry in session_sorted if entry["delta"] > 0][-5:])),
        "session_drop": [entry for entry in session_sorted if entry["delta"] < 0][:5],
    }
    return comparisons, aggregate


def _format_signed(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:+.3f}"
    return f"{value:+d}"


def _append_comparison_markdown(markdown: str, comparison: dict[str, Any]) -> str:
    lines = [markdown, "", "## Comparison vs previous run"]
    lines.append(f"- Recent mean delta: {_format_signed(comparison.get('recent_mean_delta'))}")
    lines.append(f"- Latest score delta: {_format_signed(comparison.get('latest_score_delta'))}")
    lines.append(f"- Session count delta: {_format_signed(comparison.get('session_count_delta'))}")
    model_deltas = comparison.get("model_deltas") or {}
    if model_deltas:
        lines.append("- Model deltas:")
        for key, payload in sorted(model_deltas.items()):
            session_delta = _format_signed(payload.get("session_count_delta"))
            reward_delta = _format_signed(payload.get("reward_mean_delta"))
            lines.append(f"  - {key}: sessions {session_delta}, reward_mean {reward_delta}")
    return "\n".join(lines)


def _write_outputs(
    summaries,
    output_dir: Path,
    *,
    write_markdown: bool,
    comparisons: dict[str, dict[str, Any]] | None = None,
    aggregate: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict[str, Any]] = {}
    for summary in summaries:
        key = summary.learning_key
        slug = _slug_for_key(key)
        json_path = output_dir / f"{slug}_summary.json"
        json_payload = summary_to_dict(summary)
        comparison_entry = comparisons.get(key) if comparisons else None
        if comparison_entry:
            json_payload["comparison"] = comparison_entry
        json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
        record: dict[str, Any] = {"json": str(json_path)}
        if write_markdown:
            markdown_path = output_dir / f"{slug}_summary.md"
            markdown_content = summary_to_markdown(summary)
            if comparison_entry:
                markdown_content = _append_comparison_markdown(markdown_content, comparison_entry)
            markdown_path.write_text(markdown_content, encoding="utf-8")
            record["markdown"] = str(markdown_path)
        if comparison_entry:
            record["comparison"] = comparison_entry
        manifest[key] = record
    index_path = output_dir / "index.json"
    index_payload = {
        "learning_keys": list(manifest.keys()),
        "artifacts": manifest,
    }
    if comparisons:
        index_payload["comparisons"] = comparisons
    if aggregate:
        index_payload["aggregate"] = aggregate
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
        selected_keys = args.learning_keys
        if not selected_keys:
            limit = args.limit if args.limit and args.limit > 0 else None
            selected_keys = await _load_learning_keys(
                database,
                limit=limit,
                project_root=args.filter_project,
                task=args.filter_task,
                tags=args.filter_tags,
            )
        learning_keys = list(dict.fromkeys(selected_keys))  # preserve order, remove duplicates
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
            summary_only=args.summary_only,
            project_root=args.filter_project,
            task_filter=args.filter_task,
            tags=args.filter_tags or [],
            max_concurrency=max(args.batch_size, 1),
        )
    finally:
        await database.disconnect()
    comparisons_data: dict[str, dict[str, Any]] | None = None
    aggregate_data: dict[str, Any] | None = None
    if args.compare_to:
        try:
            previous = _load_previous_summaries(args.compare_to)
        except FileNotFoundError as exc:
            raise SystemExit(f"Failed to read comparison index: {exc}") from exc
        comparisons_data, aggregate_data = _compute_comparisons(summaries, previous)
    manifest = _write_outputs(
        summaries,
        args.output_dir,
        write_markdown=not args.no_markdown,
        comparisons=comparisons_data,
        aggregate=aggregate_data,
    )
    if not args.quiet:
        print(f"Generated learning summaries for {len(manifest)} learning keys in {args.output_dir}")
        if comparisons_data:
            improved = [
                (key, data.get("recent_mean_delta"))
                for key, data in comparisons_data.items()
                if data.get("recent_mean_delta") is not None
            ]
            if improved:
                top_key, top_delta = max(improved, key=lambda item: item[1])
                print(f"Top recent-mean improvement: {top_key} ({_format_signed(top_delta)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
