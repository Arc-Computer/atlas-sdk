"""Example showing how to run Atlas with the local Python adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from atlas import core
from atlas.runtime.orchestration.execution_context import ExecutionContext


def run_agent(prompt: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Simple deterministic Python adapter implementation for demos."""
    metadata = metadata or {}
    tags = metadata.get("tags") or []
    if isinstance(tags, (list, tuple)):
        tag_summary = ", ".join(str(tag) for tag in tags) or "none"
    else:
        tag_summary = str(tags)
    return (
        "Atlas Python adapter response:\n"
        f"- Prompt: {prompt.strip() or '<empty>'}\n"
        f"- Tags: {tag_summary}"
    )


def main() -> None:
    config_path = Path("configs/examples/python_agent.yaml").resolve()
    result = core.run(
        task="Summarise the operations of the local Python agent.",
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
