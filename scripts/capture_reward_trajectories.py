from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

from atlas.config.models import RewardObjectiveConfig, RIMConfig
from atlas.core import arun as atlas_arun
from atlas.evaluation import Evaluator, SessionTrajectory, SessionStepRecord
from atlas.runtime.orchestration.execution_context import ExecutionContext
from atlas.types import Step
from atlas.utils.env import load_dotenv_if_available

DEFAULT_TASKS = Path("atlas/data/synthetic_runtime_tasks.jsonl")
DEFAULT_CONFIG = Path("configs/examples/openai_agent.yaml")
DEFAULT_COMMENT = (
    "# Atlas reward evaluation dataset. Each line is a SessionTrajectory payload captured prior to reward scoring."
)


@dataclass(slots=True)
class TaskRecord:
    task: str
    metadata: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture SessionTrajectory payloads prior to reward scoring."
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=DEFAULT_TASKS,
        help="JSONL dataset of tasks to replay (expects keys: task, metadata).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("atlas/data/reward_eval_trajectories.jsonl"),
        help="Destination JSONL file for captured trajectories.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Atlas runtime config used for execution.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Total number of trajectories to capture.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of passes over the task list (useful when limit exceeds dataset size).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle task order each repeat to add variety.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to the output file instead of replacing it.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional delay (seconds) between task executions to avoid rate limits.",
    )
    return parser.parse_args()


def load_tasks(path: Path) -> list[TaskRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Task dataset not found: {path}")
    records: list[TaskRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            task = payload.get("task")
            if not isinstance(task, str) or not task.strip():
                raise ValueError(f"Task missing or invalid on line {line_number}")
            metadata = payload.get("metadata") or {}
            if not isinstance(metadata, dict):
                raise ValueError(f"metadata must be an object on line {line_number}")
            records.append(TaskRecord(task=task.strip(), metadata=metadata))
    if not records:
        raise ValueError(f"Task dataset {path} contains no entries.")
    return records


def serialize_step(record: SessionStepRecord) -> dict[str, Any]:
    return {
        "step": record.step.model_dump(),
        "trace": record.trace,
        "output": record.output,
        "attempts": record.attempts,
        "guidance": list(record.guidance) if record.guidance else None,
        "status": record.status,
        "validation": record.validation,
        "prior_results": record.prior_results,
        "metadata": record.metadata,
    }


def serialize_trajectory(
    trajectory: SessionTrajectory,
    *,
    adaptive_summary: dict[str, Any] | None,
    session_metadata: dict[str, Any] | None,
    task_metadata: dict[str, Any] | None,
    capture_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    steps_payload = [serialize_step(step) for step in trajectory.steps]
    learning_history = session_metadata.get("learning_history") if isinstance(session_metadata, dict) else {}
    trajectory_type = "warm_start" if learning_history else "cold_start"
    return {
        "task": trajectory.task,
        "final_answer": trajectory.final_answer,
        "plan": trajectory.plan,
        "steps": steps_payload,
        "execution_mode": trajectory.execution_mode,
        "teacher_intervened": bool(trajectory.teacher_intervened),
        "adaptive_summary": adaptive_summary or {},
        "session_metadata": session_metadata or {},
        "focus_prompt": trajectory.focus_prompt,
        "trajectory_type": trajectory_type,
        "task_metadata": task_metadata or {},
        "capture_metadata": capture_metadata or {},
    }


def _normalise_payload(entry: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        return None
    entry.setdefault("steps", [])
    entry.setdefault("session_metadata", {})
    entry.setdefault("adaptive_summary", {})
    entry.setdefault("capture_metadata", {})
    entry.setdefault("task_metadata", {})
    return entry


def _is_valid_payload(entry: dict[str, Any]) -> bool:
    final_answer = str(entry.get("final_answer", "") or "").strip()
    if final_answer:
        return True
    steps = entry.get("steps") or []
    for step in steps:
        output = str((step or {}).get("output", "") or "").strip()
        if output:
            return True
    return False


def clean_metadata(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: clean_metadata(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [clean_metadata(value) for value in payload]
    if isinstance(payload, (str, int, float, bool)) or payload is None:
        return payload
    if isinstance(payload, Step):
        return payload.model_dump()
    if hasattr(payload, "model_dump"):
        try:
            return payload.model_dump()
        except Exception:
            return str(payload)
    if hasattr(payload, "__dict__"):
        return clean_metadata(vars(payload))
    return str(payload)


class TrajectoryCollector:
    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    def record(
        self,
        trajectory: SessionTrajectory,
        *,
        adaptive_summary: dict[str, Any] | None,
        session_metadata: dict[str, Any] | None,
        task_metadata: dict[str, Any] | None,
        capture_metadata: dict[str, Any] | None,
    ) -> bool:
        payload = serialize_trajectory(
            trajectory,
            adaptive_summary=adaptive_summary,
            session_metadata=session_metadata,
            task_metadata=task_metadata,
            capture_metadata=capture_metadata,
        )
        if _is_valid_payload(payload):
            self._records.append(payload)
            return True
        else:
            print(
                "[capture] skipping trajectory with empty outputs "
                f"(task='{payload.get('task', '')[:60]}')"
            )
            return False

    @property
    def records(self) -> list[dict[str, Any]]:
        return self._records

    def write(self, path: Path, *, append: bool, comment: str = DEFAULT_COMMENT) -> None:
        existing: list[dict[str, Any]] = []
        if append and path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for index, raw in enumerate(handle):
                    line = raw.rstrip("\n")
                    if index == 0 and line.startswith("#"):
                        continue
                    if not line:
                        continue
                    existing.append(json.loads(line))
        combined = [_normalise_payload(entry) for entry in existing + self._records]
        combined = [entry for entry in combined if entry is not None and _is_valid_payload(entry)]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            handle.write(comment.rstrip() + "\n")
            for entry in combined:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


class RecordingEvaluator(Evaluator):
    def __init__(
        self,
        config: RIMConfig,
        *,
        collector: TrajectoryCollector,
        task_metadata: dict[str, Any],
        capture_metadata: dict[str, Any],
        focus_prompt: str | None = None,
    ) -> None:
        super().__init__(config, focus_prompt=focus_prompt)
        self._collector = collector
        self._task_metadata = task_metadata
        self._capture_metadata = capture_metadata

    async def aevaluate_session(self, trajectory: SessionTrajectory):
        context = ExecutionContext.get()
        metadata_snapshot = clean_metadata(deepcopy(context.metadata))
        adaptive_summary = metadata_snapshot.get("adaptive_summary") or {}
        session_metadata = metadata_snapshot.get("session_metadata") or {}
        learning_history = metadata_snapshot.get("learning_history")
        if isinstance(session_metadata, dict) and not learning_history:
            learning_history = session_metadata.get("learning_history")
        if isinstance(session_metadata, dict):
            session_metadata = session_metadata
        else:
            session_metadata = {}
        if not isinstance(learning_history, dict):
            learning_history = {}
        session_metadata = dict(session_metadata)
        if learning_history:
            session_metadata.setdefault("learning_history", learning_history)
        self._collector.record(
            trajectory,
            adaptive_summary=clean_metadata(adaptive_summary),
            session_metadata=clean_metadata(session_metadata),
            task_metadata=self._task_metadata,
            capture_metadata=self._capture_metadata,
        )
        return await super().aevaluate_session(trajectory)


def build_evaluator_factory(
    collector: TrajectoryCollector,
    task_metadata: dict[str, Any],
    capture_metadata: dict[str, Any],
) -> Callable[[Any, RewardObjectiveConfig | None], Evaluator]:
    def factory(config, reward_cfg):
        reward_cfg = reward_cfg or RewardObjectiveConfig()
        if reward_cfg.type != "rim":
            raise ValueError("capture_reward_trajectories currently supports reward system only.")
        rim_config = config.rim
        if reward_cfg.parameters:
            rim_config = rim_config.model_copy(update=reward_cfg.parameters)
        focus_prompt = reward_cfg.focus_prompt or getattr(rim_config, "judge_prompt", None)
        return RecordingEvaluator(
            rim_config,
            collector=collector,
            task_metadata=task_metadata,
            capture_metadata=capture_metadata,
            focus_prompt=focus_prompt,
        )

    return factory


async def capture_trajectories(args: argparse.Namespace) -> None:
    load_dotenv_if_available()
    tasks = load_tasks(args.tasks)
    collector = TrajectoryCollector()
    total_runs = 0
    attempts = 0

    for repeat in range(args.repeats):
        if args.shuffle:
            random.shuffle(tasks)
        for index, task_entry in enumerate(tasks, start=1):
            if total_runs >= args.limit:
                break
            capture_metadata = {
                "task_index": index,
                "repeat": repeat,
                "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "task_dataset": str(args.tasks),
            }
            evaluator_factory = build_evaluator_factory(
                collector, task_entry.metadata, capture_metadata
            )
            recorded_before = len(collector.records)
            with patch("atlas.core._build_evaluator_instance", new=evaluator_factory):
                await atlas_arun(
                    task=task_entry.task,
                    config_path=str(args.config),
                    session_metadata={},
                    stream_progress=False,
                )
            attempts += 1
            recorded_after = len(collector.records)
            if recorded_after > recorded_before:
                total_runs += 1
                print(
                    f"[capture] collected {total_runs}/{args.limit} | task='{task_entry.task[:60]}'"
                )
            else:
                print(
                    f"[capture] attempt {attempts} produced empty output, retrying"
                )
            if args.sleep:
                await asyncio.sleep(args.sleep)
        if total_runs >= args.limit:
            break

    collector.write(args.output, append=args.append)
    print(f"Wrote {len(collector.records)} trajectories to {args.output}")


def main() -> None:
    args = parse_args()
    asyncio.run(capture_trajectories(args))


if __name__ == "__main__":
    main()
