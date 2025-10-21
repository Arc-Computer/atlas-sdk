"""Test doubles for the self-managed loop adapter."""

from __future__ import annotations

from atlas.connectors import EnvironmentStep, ManagedAction, ManagedObservation


class LoopEnvironment:
    instances: list["LoopEnvironment"] = []

    def __init__(self) -> None:
        self.reset_calls = 0
        self.actions: list[str] = []
        self.closed = False
        LoopEnvironment.instances.append(self)

    async def areset(self, *, task: str, metadata=None) -> ManagedObservation:
        self.reset_calls += 1
        self.last_task = task
        return ManagedObservation(content="question", metadata={"qid": metadata.get("qid") if metadata else None})

    async def astep(self, action: ManagedAction, *, metadata=None) -> EnvironmentStep:
        self.actions.append(action.command)
        done = action.submit or len(self.actions) >= 2
        observation = "final" if done else "intermediate"
        reward = 1.0 if done else 0.0
        info = {"step": len(self.actions), "metadata": metadata}
        return EnvironmentStep(observation=observation, reward=reward, done=done, info=info)

    async def aclose(self) -> None:
        self.closed = True


class LoopAgent:
    def __init__(self) -> None:
        self.event_emitter = None
        self.actions: list[ManagedAction] = []

    async def set_event_emitter(self, emitter):
        self.event_emitter = emitter

    async def areset(self, *, task: str, metadata=None) -> None:
        self.actions.clear()

    async def aplan(self, *, task: str, observation: ManagedObservation, metadata=None):
        return None

    async def aact(self, observation: str, *, step_index: int, metadata=None) -> ManagedAction:
        if step_index == 0:
            action = ManagedAction(command="query logs", submit=False, metadata={"reason": "collect context"})
        else:
            action = ManagedAction(command="submit answer", submit=True)
        self.actions.append(action)
        return action

    async def asummarize(self, trajectory, *, task: str, metadata=None) -> str:
        return "final summary"
