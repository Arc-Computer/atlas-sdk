from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from concurrent.futures import ProcessPoolExecutor, as_completed
import re
import asyncio

from atlas.config.loader import load_config
from atlas.config.models import AdapterType, AtlasConfig, LLMParameters, LLMProvider
from atlas.core import arun as atlas_arun
from atlas.runtime.orchestration.execution_context import ExecutionContext
from atlas.types import Result
from atlas.utils.env import load_dotenv_if_available

DEFAULT_DATASET = Path("atlas/data/synthetic_runtime_tasks.jsonl")
DEFAULT_CONFIG = Path("configs/examples/openai_agent.yaml")
DEFAULT_STUDENT_MODELS = ("gpt-5-mini", "claude-haiku-4-5", "gemini-2.5-flash", "grok-4-fast")
DEFAULT_TEACHER_MODELS = ("gpt-5", "claude-sonnet-4-5-20250929", "gemini-2.5-pro", "grok-4-fast")


def _disable_litellm_stream_logging() -> None:
    try:
        import litellm

        disable = getattr(litellm, "turn_off_message_logging", None)
        if callable(disable):
            disable()
    except Exception:
        pass


def _reset_litellm_logging_worker() -> None:
    try:
        from litellm.litellm_core_utils import logging_worker

        logging_worker.GLOBAL_LOGGING_WORKER = logging_worker.LoggingWorker()
    except Exception:
        pass


@dataclass(slots=True)
class RuntimeTask:
    task: str
    expected_answer: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class TaskResult:
    student_model: str
    teacher_model: str
    task: RuntimeTask
    runtime_seconds: float
    success: bool
    final_answer: str | None
    adaptive_mode: str | None
    adaptive_mode_history: list[dict[str, Any]]
    session_reward: float | None
    error: str | None


MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "gpt-5-mini": {
        "provider": LLMProvider.OPENAI,
        "model": "gpt-5-mini",
        "api_key_env": "OPENAI_API_KEY",
        "temperature": 0.2,
        "timeout_seconds": 60.0,
    },
    "gpt-5": {
        "provider": LLMProvider.OPENAI,
        "model": "gpt-5",
        "api_key_env": "OPENAI_API_KEY",
        "temperature": 0.2,
        "timeout_seconds": 60.0,
    },
    "claude-haiku-4-5": {
        "provider": LLMProvider.ANTHROPIC,
        "model": "claude-haiku-4-5",
        "api_key_env": "ANTHROPIC_API_KEY",
        "temperature": 0.2,
        "timeout_seconds": 60.0,
    },
    "claude-sonnet-4-5-20250929": {
        "provider": LLMProvider.ANTHROPIC,
        "model": "claude-sonnet-4-5-20250929",
        "api_key_env": "ANTHROPIC_API_KEY",
        "temperature": 0.15,
        "timeout_seconds": 60.0,
    },
    "gemini-2.5-flash": {
        "provider": LLMProvider.GEMINI,
        "model": "gemini/gemini-2.5-flash",
        "api_key_env": "GEMINI_API_KEY",
        "temperature": 0.2,
        "timeout_seconds": 60.0,
    },
    "gemini-2.5-pro": {
        "provider": LLMProvider.GEMINI,
        "model": "gemini/gemini-2.5-pro",
        "api_key_env": "GEMINI_API_KEY",
        "temperature": 0.2,
        "timeout_seconds": 60.0,
    },
    "grok-4-fast": {
        "provider": LLMProvider.XAI,
        "model": "xai/grok-4-fast",
        "api_key_env": "XAI_API_KEY",
        "temperature": 0.2,
        "timeout_seconds": 45.0,
    },
}


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate dual-agent model pairings on synthetic runtime tasks.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Path to JSONL dataset.")
    parser.add_argument(
        "--base-config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Base Atlas runtime config used for overrides.",
    )
    parser.add_argument(
        "--student-models",
        nargs="+",
        default=list(DEFAULT_STUDENT_MODELS),
        help="Student model identifiers.",
    )
    parser.add_argument(
        "--teacher-models",
        nargs="+",
        default=list(DEFAULT_TEACHER_MODELS),
        help="Teacher model identifiers.",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Number of times to repeat each task per model pair.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum number of concurrent task executions (sequential by default).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write JSON results containing summaries and per-run details.",
    )
    return parser.parse_args(argv)


def load_dataset(path: Path) -> list[RuntimeTask]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    tasks: list[RuntimeTask] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            task = str(payload.get("task", "")).strip()
            expected = str(payload.get("expected_answer", "")).strip()
            metadata = payload.get("metadata") or {}
            if not task or not expected:
                raise ValueError(f"Dataset entry missing task or expected_answer on line {line_number}")
            if not isinstance(metadata, dict):
                raise ValueError(f"metadata must be an object on line {line_number}")
            tasks.append(RuntimeTask(task=task, expected_answer=expected, metadata=metadata))
    if not tasks:
        raise ValueError(f"Dataset {path} contained no tasks.")
    return tasks


def build_llm_parameters(
    model_id: str,
    *,
    role: str,
    adapter_type: AdapterType | None = None,
) -> LLMParameters:
    preset = MODEL_PRESETS.get(model_id)
    if preset is None:
        supported = ", ".join(sorted(MODEL_PRESETS))
        raise ValueError(f"Unknown model '{model_id}'. Supported models: {supported}")
    temperature = preset["temperature"]
    if role == "teacher":
        temperature = min(temperature, 0.15)
    env_key_suffix = re.sub(r"[^A-Z0-9]+", "_", model_id.upper()).strip("_")
    override_key = f"ATLAS_MODEL_OVERRIDE_{env_key_suffix}"
    provider: LLMProvider = preset["provider"]
    if adapter_type == AdapterType.OPENAI and role == "student":
        provider = LLMProvider.OPENAI
    return LLMParameters(
        provider=provider,
        model=os.environ.get(override_key) or preset["model"],
        api_key_env=preset["api_key_env"],
        temperature=temperature,
        timeout_seconds=float(os.environ.get("ATLAS_MODEL_TIMEOUT", preset["timeout_seconds"])),
    )


def override_config(
    base_config: AtlasConfig,
    *,
    student_model: str,
    teacher_model: str,
) -> AtlasConfig:
    student_params = build_llm_parameters(student_model, role="student", adapter_type=base_config.agent.type)
    teacher_params = build_llm_parameters(teacher_model, role="teacher")
    return base_config.model_copy(
        update={
            "agent": base_config.agent.model_copy(update={"llm": student_params}),
            "teacher": base_config.teacher.model_copy(update={"llm": teacher_params}),
        },
        deep=True,
    )


def write_config_copy(config: AtlasConfig, directory: Path, student_model: str, teacher_model: str) -> Path:
    payload = config.model_dump(mode="json")
    filename = f"runtime_{student_model}_{teacher_model}.json"
    target = directory / filename
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


async def _arun_with_metadata(task: RuntimeTask, config_path: Path) -> tuple[Result, dict[str, Any]]:
    execution_context = ExecutionContext.get()
    execution_context.reset()
    result = await atlas_arun(
        task.task,
        str(config_path),
        stream_progress=False,
    )
    metadata = dict(execution_context.metadata)
    execution_context.reset()
    return result, metadata


def execute_task(task: RuntimeTask, config_path: Path) -> tuple[Result, dict[str, Any]]:
    return asyncio.run(_arun_with_metadata(task, config_path))


def build_task_result(
    *,
    student_model: str,
    teacher_model: str,
    task: RuntimeTask,
    result: Result | None,
    metadata: dict[str, Any] | None,
    runtime_seconds: float,
    error: str | None,
) -> TaskResult:
    metadata = metadata or {}
    adaptive_summary = metadata.get("adaptive_summary") if isinstance(metadata, dict) else None
    adaptive_mode = None
    adaptive_history: list[dict[str, Any]] = []
    if isinstance(adaptive_summary, dict):
        adaptive_mode = adaptive_summary.get("adaptive_mode")
        history = adaptive_summary.get("mode_history")
        if isinstance(history, list):
            adaptive_history = [entry for entry in history if isinstance(entry, dict)]
    session_reward_payload = metadata.get("session_reward") if isinstance(metadata, dict) else None
    reward_score: float | None = None
    if isinstance(session_reward_payload, dict):
        score = session_reward_payload.get("score")
        if isinstance(score, (int, float)):
            reward_score = float(score)
    final_answer = result.final_answer if result is not None else None
    success = error is None
    return TaskResult(
        student_model=student_model,
        teacher_model=teacher_model,
        task=task,
        runtime_seconds=runtime_seconds,
        success=success,
        final_answer=final_answer,
        adaptive_mode=adaptive_mode,
        adaptive_mode_history=adaptive_history,
        session_reward=reward_score,
        error=error,
    )


def aggregate_results(results: Iterable[TaskResult]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[TaskResult]] = defaultdict(list)
    for record in results:
        grouped[(record.student_model, record.teacher_model)].append(record)

    summaries: list[dict[str, Any]] = []
    for (student_model, teacher_model), records in grouped.items():
        successes = [rec for rec in records if rec.success]
        runtime_values = [rec.runtime_seconds for rec in successes]
        reward_values = [rec.session_reward for rec in successes if rec.session_reward is not None]
        adaptive_counter = Counter(rec.adaptive_mode for rec in successes if rec.adaptive_mode)

        average_runtime = statistics.mean(runtime_values) if runtime_values else None
        average_reward = statistics.mean(reward_values) if reward_values else None
        summaries.append(
            {
                "student_model": student_model,
                "teacher_model": teacher_model,
                "total_runs": len(records),
                "failures": len(records) - len(successes),
                "average_runtime_seconds": average_runtime,
                "average_reward": average_reward,
                "adaptive_modes": dict(adaptive_counter),
            }
        )
    def _sort_key(item: dict[str, Any]) -> tuple[float, float]:
        reward = item["average_reward"]
        avg_runtime = item["average_runtime_seconds"] or float("inf")
        return (-(reward if reward is not None else float("-inf")), avg_runtime)

    return sorted(summaries, key=_sort_key)


def print_summary_table(summaries: Sequence[dict[str, Any]]) -> None:
    if not summaries:
        print("No results to display.", file=sys.stderr)
        return
    headers = (
        "Student",
        "Teacher",
        "Avg Reward",
        "Avg Runtime (s)",
        "Failures",
        "Modes",
    )
    rows: list[tuple[str, ...]] = []
    for entry in summaries:
        avg_reward = entry["average_reward"]
        avg_runtime = entry["average_runtime_seconds"]
        modes = " ".join(f"{mode}:{count}" for mode, count in sorted(entry["adaptive_modes"].items()))
        rows.append(
            (
                entry["student_model"],
                entry["teacher_model"],
                f"{avg_reward:.2f}" if avg_reward is not None else "n/a",
                f"{avg_runtime:.2f}" if avg_runtime is not None else "n/a",
                str(entry["failures"]),
                modes or "-",
            )
        )
    widths = [max(len(header), *(len(row[idx]) for row in rows)) for idx, header in enumerate(headers)]
    header_line = " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))
    divider = "-+-".join("-" * widths[idx] for idx in range(len(headers)))
    print(header_line)
    print(divider)
    for row in rows:
        print(" | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))))


def select_best_pair(summaries: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not summaries:
        return None
    top = summaries[0]
    return {
        "student_model": top["student_model"],
        "teacher_model": top["teacher_model"],
        "average_runtime_seconds": top["average_runtime_seconds"],
        "average_reward": top["average_reward"],
        "failures": top["failures"],
    }


def ensure_models_exist(models: Iterable[str]) -> None:
    unsupported = [model for model in models if model not in MODEL_PRESETS]
    if unsupported:
        supported = ", ".join(sorted(MODEL_PRESETS))
        raise ValueError(f"Unsupported model identifiers: {', '.join(unsupported)}. Supported models: {supported}")


def _execute_job(
    student_model: str,
    teacher_model: str,
    task: RuntimeTask,
    config_path: str,
) -> TaskResult:
    _reset_litellm_logging_worker()
    _disable_litellm_stream_logging()
    start = time.perf_counter()
    error: str | None = None
    result: Result | None = None
    metadata: dict[str, Any] | None = None
    try:
        result, metadata = execute_task(task, Path(config_path))
    except Exception as exc:  # pragma: no cover - defensive guard for runtime errors
        error = str(exc)
    runtime_seconds = time.perf_counter() - start
    record = build_task_result(
        student_model=student_model,
        teacher_model=teacher_model,
        task=task,
        result=result,
        metadata=metadata,
        runtime_seconds=runtime_seconds,
        error=error,
    )
    status = "ok" if error is None else f"error: {error}"
    print(
        f"[eval] Completed student={student_model} teacher={teacher_model} "
        f"task=\"{task.task[:48]}\" status={status} runtime={runtime_seconds:.2f}s",
        flush=True,
    )
    return record


def run_evaluations(args: argparse.Namespace) -> tuple[list[TaskResult], list[dict[str, Any]]]:
    load_dotenv_if_available()
    ensure_models_exist(args.student_models)
    ensure_models_exist(args.teacher_models)

    dataset = load_dataset(args.dataset)
    base_config = load_config(args.base_config)

    task_results: list[TaskResult] = []
    with tempfile.TemporaryDirectory(prefix="atlas-eval-configs-") as tmp_dir:
        config_cache: dict[tuple[str, str], Path] = {}
        temp_dir = Path(tmp_dir)
        for student_model in args.student_models:
            for teacher_model in args.teacher_models:
                key = (student_model, teacher_model)
                overridden_config = override_config(
                    base_config,
                    student_model=student_model,
                    teacher_model=teacher_model,
                )
                config_cache[key] = write_config_copy(overridden_config, temp_dir, student_model, teacher_model)

        jobs: list[tuple[str, str, RuntimeTask, str]] = []
        for student_model in args.student_models:
            for teacher_model in args.teacher_models:
                config_path = config_cache[(student_model, teacher_model)]
                for _ in range(args.repeats):
                    for task in dataset:
                        jobs.append((student_model, teacher_model, task, str(config_path)))

        if args.concurrency <= 1:
            for student_model, teacher_model, task, config_path in jobs:
                task_results.append(
                    _execute_job(
                        student_model=student_model,
                        teacher_model=teacher_model,
                        task=task,
                        config_path=config_path,
                    )
                )
        else:
            with ProcessPoolExecutor(max_workers=args.concurrency) as executor:
                futures = [
                    executor.submit(
                        _execute_job,
                        student_model,
                        teacher_model,
                        task,
                        config_path,
                    )
                    for student_model, teacher_model, task, config_path in jobs
                ]
                for future in as_completed(futures):
                    task_results.append(future.result())

    summaries = aggregate_results(task_results)
    return task_results, summaries


def write_output(
    output_path: Path,
    *,
    dataset_path: Path,
    args: argparse.Namespace,
    task_results: Sequence[TaskResult],
    summaries: Sequence[dict[str, Any]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": str(dataset_path),
        "student_models": list(args.student_models),
        "teacher_models": list(args.teacher_models),
        "repeats": args.repeats,
        "summaries": summaries,
        "best_pair": select_best_pair(summaries),
        "runs": [
            {
                "student_model": record.student_model,
                "teacher_model": record.teacher_model,
                "task": record.task.task,
                "expected_answer": record.task.expected_answer,
                "metadata": record.task.metadata,
                "runtime_seconds": record.runtime_seconds,
                "success": record.success,
                "final_answer": record.final_answer,
                "adaptive_mode": record.adaptive_mode,
                "adaptive_mode_history": record.adaptive_mode_history,
                "session_reward": record.session_reward,
                "error": record.error,
            }
            for record in task_results
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        task_results, summaries = run_evaluations(args)
    except Exception as exc:
        print(f"Error while running evaluations: {exc}", file=sys.stderr)
        return 1
    print_summary_table(summaries)
    if args.output:
        write_output(args.output, dataset_path=args.dataset, args=args, task_results=task_results, summaries=summaries)
        print(f"\nWrote results to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
def _disable_litellm_stream_logging() -> None:
    try:
        import litellm

        disable = getattr(litellm, "disable_streaming_logging", None)
        if callable(disable):
            disable()
    except Exception:
        pass
