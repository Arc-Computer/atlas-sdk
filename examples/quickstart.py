"""Minimal Atlas quickstart demonstrating two learning passes."""

from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message='Field name "schema" in "LearningConfig" shadows an attribute in parent "BaseModel"',
    category=UserWarning,
)

import asyncio
import logging
import os
import time
from typing import Any, Dict, Tuple

from atlas import core
from atlas.cli.run import _render_learning_summary
from atlas.runtime.orchestration.execution_context import ExecutionContext
from atlas.utils.env import load_dotenv_if_available

load_dotenv_if_available()

# Task intentionally avoids tool calls because some configs lack optional tool registrations.
TASK = (
    "Review the project memo below and produce a three-item checklist for a new "
    "Atlas SDK contributor. Each checklist item must start with an imperative "
    "verb, stay under 12 words, and avoid requesting additional tools or data.\n\n"
    "Project memo:\n"
    "- Install dependencies with `pip install -e .`.\n"
    "- Explore quickstart scripts in `examples/`.\n"
    "- Run `pytest` before opening a pull request.\n"
)
DEFAULT_CONFIG_PATH = "configs/examples/openai_agent.yaml"
CONFIG_OVERRIDE_ENV = "ATLAS_QUICKSTART_CONFIG"
PASS_HEADERS = [
    "=== PASS 1: Learning Phase ===",
    "=== PASS 2: Applied Learning ===",
]

GLOBAL_LOGGING_WORKER: Any | None = None


def _resolve_config_path() -> str:
    config_path = os.environ.get(CONFIG_OVERRIDE_ENV, DEFAULT_CONFIG_PATH)
    if not os.path.exists(config_path):
        print(
            f"Quickstart error: config path '{config_path}' not found. "
            f"Set {CONFIG_OVERRIDE_ENV} or update DEFAULT_CONFIG_PATH.",
        )
        raise SystemExit(1)
    return config_path


def _ensure_api_keys() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        logging.error(
            "Quickstart needs OPENAI_API_KEY to run; export it or set %s to a custom config.",
            CONFIG_OVERRIDE_ENV,
        )
        raise SystemExit(1)


async def _run_pass(header: str, config_path: str) -> Tuple[Any, Dict[str, Any], float]:
    print(header)
    start = time.perf_counter()
    try:
        result = await core.arun(task=TASK, config_path=config_path, stream_progress=True)
    except Exception as exc:  # pragma: no cover - defensive guard for quickstart clarity
        logging.error(
            "Quickstart failed: %s. Check API keys or switch task/config to one that doesn't "
            "require tools.",
            exc,
        )
        raise SystemExit(1) from exc
    duration = time.perf_counter() - start
    metadata = dict(ExecutionContext.get().metadata)
    return result, metadata, duration


async def _shutdown_litellm_worker() -> None:
    if GLOBAL_LOGGING_WORKER is None:
        return
    try:
        await GLOBAL_LOGGING_WORKER.stop()
    except Exception:  # pragma: no cover - best effort cleanup
        logging.getLogger(__name__).debug("LiteLLM logging worker stop failed", exc_info=True)


async def _main_async() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("Orchestrator").setLevel(logging.WARNING)
    config_path = _resolve_config_path()
    _ensure_api_keys()

    global GLOBAL_LOGGING_WORKER
    if GLOBAL_LOGGING_WORKER is None:
        try:  # pragma: no cover - optional dependency at runtime
            from litellm.litellm_core_utils.logging_worker import (
                GLOBAL_LOGGING_WORKER as litellm_worker,
            )
        except Exception:  # pragma: no cover - litellm not installed or misconfigured
            litellm_worker = None
        GLOBAL_LOGGING_WORKER = litellm_worker

    try:
        for index, header in enumerate(PASS_HEADERS, start=1):
            if index > 1:
                print()
            result, metadata, duration = await _run_pass(header, config_path)
            pass_label = f"Pass {index}"
            print(f"\n--- {pass_label} Final Answer ---")
            print(result.final_answer)
            print(f"\n--- {pass_label} Session Summary ---")
            print(f"{pass_label} Duration: {duration:.1f}s")
            summary_text = _render_learning_summary(metadata, stream=True)
            if summary_text:
                print(summary_text)
            else:
                print("No learning telemetry captured (check config's learning settings).")

        print(f"Completed quickstart using {config_path}")
        print(
            "What you just saw: Atlas ran the dual-pass loop, produced final answers, and "
            "streamed learning telemetry from the default config."
        )
        print(
            "Next steps: tweak TASK in examples/quickstart.py, or follow README.md to wire "
            "Atlas into your own agent stack."
        )
    finally:
        await _shutdown_litellm_worker()


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Quickstart interrupted by user.")
