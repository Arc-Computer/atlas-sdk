"""Run Atlas GDPval demo tasks with telemetry and persistence."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from atlas.core import arun
from atlas.dashboard import TelemetryPublisher

from .agent import build_session_metadata
from .loader import CACHE_ROOT
from .loader import GDPValTask
from .loader import load_gdpval_tasks
from .loader import ensure_manifest


DEFAULT_CONFIG = "configs/examples/gdpval_python.yaml"
SUMMARY_DIR = Path("examples/gdpval").resolve()


def main(argv: Optional[List[str]] = None) -> None:
    _load_env_file()
    parser = build_parser()
    args = parser.parse_args(argv)
    asyncio.run(run_cli(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Atlas GDPval demo tasks")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task-id", help="Run a single GDPval task by identifier")
    group.add_argument("--all", action="store_true", help="Iterate every task in the split")
    parser.add_argument("--max", type=int, default=None, help="Maximum number of tasks to execute when using --all")
    parser.add_argument("--split", default="train", help="Dataset split to use (default: train)")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Atlas configuration file")
    parser.add_argument("--streaming", action="store_true", help="Use streaming dataset mode")
    parser.add_argument("--summary", default=str(SUMMARY_DIR / "gdpval_runs"), help="Directory to persist summaries")
    return parser


async def run_cli(args: argparse.Namespace) -> None:
    tasks = _resolve_tasks(args)
    summary_dir = Path(args.summary)
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / "runs.csv"
    json_path = summary_dir / "runs.jsonl"
    results: List[dict] = []
    count = 0
    publisher = TelemetryPublisher()
    for task in tasks:
        if args.max is not None and count >= args.max:
            break
        try:
            record = await _run_task(
                task=task,
                config_path=args.config,
                publisher=publisher,
            )
            results.append(record)
            preview = (record.get("final_answer") or "").strip().replace("\n", " ")
            if preview:
                preview = preview[:120] + ("…" if len(preview) > 120 else "")
            print(f"[GDPval] {record['task_id']} | attempts={record['attempts']} | rim_score={record['rim_score']} | {preview}")
        except Exception as exc:  # noqa: BLE001
            warning = {
                "task_id": task.task_id,
                "error": str(exc),
            }
            results.append(warning)
            print(f"[GDPval] {task.task_id} failed: {exc}")
        count += 1
    _write_csv(csv_path, results)
    _write_jsonl(json_path, results)
    print(f"[GDPval] wrote summaries to {csv_path} and {json_path}")


def _resolve_tasks(args: argparse.Namespace) -> Iterable[GDPValTask]:
    if args.task_id:
        alias_prefix = "gdpval_task_"
        if args.task_id.startswith(alias_prefix):
            index_str = args.task_id[len(alias_prefix) :]
            try:
                target_index = int(index_str) - 1
            except ValueError:
                raise RuntimeError(f"Unrecognised task alias {args.task_id}") from None
            if target_index < 0:
                raise RuntimeError(f"Task alias {args.task_id} is out of range")
            for idx, task in enumerate(
                load_gdpval_tasks(split=args.split, streaming=False, cache_references=True)
            ):
                if idx == target_index:
                    return [task]
            raise RuntimeError(f"Task alias {args.task_id} is out of range for split {args.split}")
        for task in load_gdpval_tasks(split=args.split, streaming=False, cache_references=True):
            if task.task_id == args.task_id:
                return [task]
        raise RuntimeError(f"Task {args.task_id} not found in GDPval split {args.split}")
    iterator = load_gdpval_tasks(split=args.split, streaming=args.streaming, cache_references=False)
    return iterator


async def _run_task(*, task: GDPValTask, config_path: str, publisher: TelemetryPublisher) -> dict:
    for reference in task.references:
        reference.cache(task.task_id)
    ensure_manifest(task)
    manifest = task.to_manifest()
    session_metadata = build_session_metadata(task)
    prompt = _compose_prompt(task)
    summary = await arun(
        prompt,
        config_path,
        publisher=publisher,
        session_metadata=session_metadata,
    )
    rim_scores = []
    for step in summary.step_results:
        evaluation = step.evaluation or {}
        reward = evaluation.get("reward") if isinstance(evaluation, dict) else None
        score = reward.get("score") if isinstance(reward, dict) else None
        if isinstance(score, (int, float)):
            rim_scores.append(score)
    top_score = max(rim_scores, default=None)
    return {
        "task_id": task.task_id,
        "sector": task.sector,
        "occupation": task.occupation,
        "final_answer": summary.final_answer,
        "manifest": manifest,
        "rim_score": top_score,
        "attempts": len(summary.step_results),
    }


def _compose_prompt(task: GDPValTask) -> str:
    header = (
        f"GDPval Task ID: {task.task_id}\n"
        f"Sector: {task.sector}\n"
        f"Occupation: {task.occupation}\n"
        "Use the cached GDPval references to justify every claim.\n\n"
    )
    return header + task.prompt


def _write_csv(path: Path, records: List[dict]) -> None:
    fieldnames = ["task_id", "sector", "occupation", "final_answer", "rim_score", "attempts", "error"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fieldnames})


def _write_jsonl(path: Path, records: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")


def _load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if sep != "=":
            continue
        key = key.strip()
        if not key or key.startswith("#"):
            continue
        value = value.strip().strip('"')
        os.environ.setdefault(key, value)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
