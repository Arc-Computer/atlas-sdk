"""Example showing how to run Atlas with the HTTP adapter configuration."""

from __future__ import annotations

from pathlib import Path

from atlas import core
from atlas.runtime.orchestration.execution_context import ExecutionContext


def main() -> None:
    config_path = Path("configs/examples/http_agent.yaml").resolve()
    result = core.run(
        task="Query the HTTP agent for the latest status update.",
        config_path=str(config_path),
        stream_progress=True,
    )

    print("\n=== Final Answer ===")
    print(result.final_answer)

    metadata = ExecutionContext.get().metadata
    adaptive = metadata.get("adaptive_summary")
    if adaptive:
        print("\nAdaptive summary:", adaptive)
    reward = metadata.get("reward_summary") or metadata.get("session_reward")
    if reward:
        print("Reward summary:", reward)


if __name__ == "__main__":
    main()
