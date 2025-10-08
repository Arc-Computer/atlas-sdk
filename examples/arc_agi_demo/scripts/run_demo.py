"""Run the ARC-AGI demo workflow using pip-installed arc-atlas."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None  # type: ignore

if load_dotenv is not None:
    load_dotenv()

from atlas import core

from examples.arc_agi_demo.scripts.task_utils import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_TASK_PATH,
    arc_task_to_prompt,
    load_arc_task,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ARC-AGI demo with arc-atlas.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the Atlas configuration file.",
    )
    parser.add_argument(
        "--task-id",
        default=None,
        help="ARC training task identifier to download from the official dataset.",
    )
    parser.add_argument(
        "--task-file",
        type=Path,
        help="Local ARC task JSON file. Overrides --task-id when provided.",
    )
    parser.add_argument(
        "--stream-progress",
        action="store_true",
        help="Force console telemetry streaming even when stdout is not a TTY.",
    )
    return parser.parse_args()


def _render_plan(plan_steps: List[str]) -> str:
    return "\n".join(f"{index+1}. {step}" for index, step in enumerate(plan_steps))


def main() -> None:
    args = _parse_args()
    task_path = args.task_file
    if not task_path and not args.task_id:
        task_path = DEFAULT_TASK_PATH
    payload = load_arc_task(args.task_id, task_path)
    prompt = arc_task_to_prompt(payload)
    result = core.run(
        task=prompt,
        config_path=str(args.config),
        stream_progress=args.stream_progress,
    )
    print("\n=== Review Plan ===")
    print(_render_plan([step.description for step in result.plan.steps]))
    print("\n=== Final Answer ===")
    print(result.final_answer)


if __name__ == "__main__":
    main()
