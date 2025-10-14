"""Minimal Atlas quickstart showing the lightweight default workflow."""

from __future__ import annotations

import logging

from atlas import core
from atlas.runtime.orchestration.execution_context import ExecutionContext


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = core.run(
        task="Summarise the latest Atlas SDK updates in three bullet points.",
        config_path="configs/examples/openai_agent.yaml",
        stream_progress=True,
    )

    print("\n=== Final Answer ===")
    print(result.final_answer)

    metadata = ExecutionContext.get().metadata
    adaptive = metadata.get("adaptive_summary")
    if isinstance(adaptive, dict):
        print("\nAdaptive summary:", adaptive)
    reward = metadata.get("reward_summary")
    if isinstance(reward, dict):
        print("Reward summary:", reward)
    else:
        session_reward = metadata.get("session_reward")
        if isinstance(session_reward, dict):
            print("Reward summary:", session_reward)


if __name__ == "__main__":
    main()
