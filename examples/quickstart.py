"""Minimal Atlas quickstart demonstrating two learning passes."""

from __future__ import annotations

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

TASK = "Summarize the latest Atlas SDK updates in three bullet points."
DEFAULT_CONFIG_PATH = "configs/examples/openai_agent.yaml"
CONFIG_OVERRIDE_ENV = "ATLAS_QUICKSTART_CONFIG"
PASS_HEADERS = [
    "=== PASS 1: Learning Phase ===",
    "=== PASS 2: Applied Learning ===",
]

try:  # pragma: no cover - optional dependency at runtime
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
except Exception:  # pragma: no cover - defensive guard
    GLOBAL_LOGGING_WORKER = None  # type: ignore[assignment]


def _resolve_config_path() -> str:
    return os.environ.get(CONFIG_OVERRIDE_ENV, DEFAULT_CONFIG_PATH)
async def _run_pass(header: str, config_path: str) -> Tuple[Any, Dict[str, Any], float]:
    print(header)
    start = time.perf_counter()
    result = await core.arun(task=TASK, config_path=config_path, stream_progress=True)
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
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config_path = _resolve_config_path()

    for index, header in enumerate(PASS_HEADERS):
        if index:
            print()
        result, metadata, duration = await _run_pass(header, config_path)
        print("\n--- Final Answer ---")
        print(result.final_answer)
        print("\n--- Session Summary ---")
        print(f"Duration: {duration:.1f}s")
        summary_text = _render_learning_summary(metadata, stream=True)
        if summary_text:
            print(summary_text)
        else:
            print("Learning summary unavailable.")

    await _shutdown_litellm_worker()


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
