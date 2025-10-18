"""Minimal Atlas quickstart demonstrating two learning passes."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, Tuple

from atlas import core
from atlas.runtime.orchestration.execution_context import ExecutionContext

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


def _summarize_reward(metadata: Dict[str, Any]) -> str:
    reward = metadata.get("reward_summary") or metadata.get("session_reward")
    if isinstance(reward, dict):
        score = reward.get("score")
        if isinstance(score, (int, float)):
            return f"Reward score: {score:.2f}"
    return "Reward score: n/a"


def _summarize_tokens(metadata: Dict[str, Any]) -> str:
    usage = metadata.get("token_usage")
    if not isinstance(usage, dict):
        return "Token usage: n/a"
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    total = usage.get("total_tokens")
    if all(isinstance(value, (int, float)) for value in (prompt, completion, total)):
        return (
            "Token usage: "
            f"prompt={int(prompt)} completion={int(completion)} total={int(total)}"
        )
    return "Token usage: n/a"


def _summarize_adaptive(metadata: Dict[str, Any]) -> str:
    adaptive = metadata.get("adaptive_summary")
    if isinstance(adaptive, dict):
        mode = adaptive.get("adaptive_mode")
        confidence = adaptive.get("confidence")
        if mode and isinstance(confidence, (int, float)):
            return f"Adaptive mode: {mode} (confidence={confidence:.2f})"
        if mode:
            return f"Adaptive mode: {mode}"
    mode = metadata.get("execution_mode")
    if mode:
        return f"Adaptive mode: {mode}"
    return "Adaptive mode: n/a"


def _print_pass_summary(result: Any, metadata: Dict[str, Any], duration: float) -> None:
    print("\n--- Final Answer ---")
    print(result.final_answer)
    print("\n--- Session Summary ---")
    print(f"Duration: {duration:.1f}s")
    print(_summarize_reward(metadata))
    print(_summarize_tokens(metadata))
    print(_summarize_adaptive(metadata))
    learning_key = metadata.get("learning_key")
    if isinstance(learning_key, str):
        print(f"Learning key: {learning_key[:32]}...")


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
        _print_pass_summary(result, metadata, duration)

    await _shutdown_litellm_worker()


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
