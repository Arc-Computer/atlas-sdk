"""Sample stateful environment/agent pairing for Atlas autodiscovery."""

from __future__ import annotations

from atlas import agent, environment
from atlas.sdk.interfaces import DiscoveryContext, TelemetryEmitterProtocol


@environment
class CounterEnvironment:
    """Minimal environment that increments a counter until the agent stops."""

    def __init__(self) -> None:
        self._count = 0

    def reset(self, task: str | None = None) -> dict[str, int | str | None]:
        self._count = 0
        return {"count": self._count, "task": task}

    def step(self, action: dict[str, int] | None):
        increment = 1
        if isinstance(action, dict):
            increment = int(action.get("delta", 1)) or 1
        self._count += increment
        observation = {"count": self._count}
        done = self._count >= 3
        reward = 1.0 if done else 0.0
        info = {"increment": increment}
        return observation, reward, done, info

    def close(self) -> None:  # pragma: no cover - no cleanup required
        return None


@agent
class CounterAgent:
    """Simple agent that increments the counter and emits telemetry."""

    def plan(
        self,
        task: str,
        observation: dict[str, int | str | None],
        *,
        emit_event: TelemetryEmitterProtocol | None = None,
    ) -> dict[str, str]:
        if emit_event:
            emit_event.emit("progress", {"stage": "plan", "task": task})
        return {"goal": "reach a count of three"}

    def act(
        self,
        context: DiscoveryContext,
        *,
        emit_event: TelemetryEmitterProtocol | None = None,
    ) -> dict[str, int]:
        if emit_event:
            emit_event.emit("progress", {"stage": "act", "step": context.step_index})
        return {"delta": 1}

    def summarize(
        self,
        context: DiscoveryContext,
        *,
        history=None,
        emit_event: TelemetryEmitterProtocol | None = None,
    ) -> str:
        if emit_event:
            emit_event.emit("progress", {"stage": "summarize", "total_steps": context.step_index})
        return "Counter reached three."
