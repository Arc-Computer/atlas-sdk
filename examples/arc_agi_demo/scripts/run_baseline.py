"""Invoke the demo agent directly without Atlas orchestration."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore

if load_dotenv is not None:
    load_dotenv()

from examples.arc_agi_demo.agent.demo_agent import invoke
from examples.arc_agi_demo.scripts.task_utils import (
    DEFAULT_TASK_PATH,
    arc_task_to_prompt,
    load_arc_task,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ARC demo agent without Atlas.")
    parser.add_argument(
        "--task-id",
        default=None,
        help="ARC training task identifier to download.",
    )
    parser.add_argument(
        "--task-file",
        type=Path,
        help="Local ARC task JSON file to load instead of downloading.",
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> None:
    task_path = args.task_file
    if not task_path and not args.task_id:
        task_path = DEFAULT_TASK_PATH
    payload = load_arc_task(args.task_id, task_path)
    prompt = arc_task_to_prompt(payload)
    result = await invoke(prompt, metadata={"mode": "baseline"})
    print("=== Baseline Agent Output ===")
    print(result)


def main() -> None:
    args = _parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
