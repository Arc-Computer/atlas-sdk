from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Tuple

import pytest


@pytest.fixture()
def stateful_project(tmp_path: Path) -> Tuple[Path, str, str, str]:
    source = textwrap.dedent(
        """
        from __future__ import annotations

        from atlas import agent, environment
        from atlas.sdk.interfaces import DiscoveryContext, TelemetryEmitterProtocol


        @environment
        class DemoEnvironment:
            def __init__(self) -> None:
                self._step = 0

            def reset(self, task: str | None = None):
                self._step = 0
                return {"count": 0, "task": task}

            def step(self, action):
                self._step += int(action.get("delta", 1)) if isinstance(action, dict) else 1
                observation = {"count": self._step}
                done = self._step >= 2
                reward = 1.0 if done else 0.5
                return observation, reward, done, {"step": self._step}

            def close(self) -> None:  # pragma: no cover - no-op cleanup
                return None


        @agent
        class DemoAgent:
            def plan(
                self,
                task: str,
                observation,
                *,
                emit_event: TelemetryEmitterProtocol | None = None,
            ):
                if emit_event:
                    emit_event.emit("progress", {"phase": "plan", "task": task})
                return {"goal": "increment until done"}

            def act(
                self,
                context: DiscoveryContext,
                *,
                emit_event: TelemetryEmitterProtocol | None = None,
            ):
                if emit_event:
                    emit_event.emit("progress", {"phase": "act", "step": context.step_index})
                return {"delta": 1}

            def summarize(
                self,
                context: DiscoveryContext,
                *,
                history=None,
                emit_event: TelemetryEmitterProtocol | None = None,
            ) -> str:
                if emit_event:
                    emit_event.emit("progress", {"phase": "summarize", "steps": context.step_index})
                return "Completed increments"
        """
    )
    module_path = tmp_path / "demo.py"
    module_path.write_text(source, encoding="utf-8")
    return tmp_path, "demo", "DemoEnvironment", "DemoAgent"
