"""Bootstrap helpers illustrating how to wrap SecRL-style environments for Atlas."""

from __future__ import annotations

from atlas.sdk.interfaces import DiscoveryContext, TelemetryEmitterProtocol


class MockSecGymEnv:
    """Minimal SecRL-style environment that requires constructor hints."""

    def __init__(self, attack: str, db_url: str):
        self._attack = attack
        self._db_url = db_url
        self._counter = 0

    def reset(self, task: str | None = None):
        self._counter = 0
        return {"question": f"Investigate {self._attack}", "db": self._db_url}

    def step(self, action: str, submit: bool = False):
        self._counter += 1
        if submit:
            return "", 1.0, True, {"submitted": action}
        return f"log chunk {self._counter}", 0.0, False, {"query": action}

    def close(self) -> None:
        return None


class MockSecGymAgent:
    """Agent that exposes only act/reset—Atlas will wrap plan/summarize automatically."""

    def __init__(self):
        self._submitted = False

    def reset(self) -> None:
        self._submitted = False

    def act(self, observation):
        if self._submitted:
            return ("submit[Escalate to human]", True)
        self._submitted = True
        return ("SELECT * FROM alerts LIMIT 1", False)


def create_environment(attack: str, db_url: str) -> MockSecGymEnv:
    """Factory passed to `atlas env init --env-fn`."""

    return MockSecGymEnv(attack=attack, db_url=db_url)


def create_agent() -> MockSecGymAgent:
    """Factory passed to `atlas env init --agent-fn`."""

    return MockSecGymAgent()
