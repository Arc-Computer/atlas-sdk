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


@pytest.fixture()
def secrl_project(tmp_path: Path) -> Tuple[Path, str, str, str]:
    source = textwrap.dedent(
        """
        from __future__ import annotations

        from atlas.sdk.interfaces import DiscoveryContext, TelemetryEmitterProtocol


        class MockSecGymEnv:
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

            def close(self):
                return None


        class MockSecGymAgent:
            def __init__(self):
                self._submitted = False

            def reset(self):
                self._submitted = False

            def act(self, observation):
                if self._submitted:
                    return ("submit[Escalate to human]", True)
                self._submitted = True
                return ("SELECT * FROM alerts LIMIT 1", False)


        def create_environment(attack: str, db_url: str):
            return MockSecGymEnv(attack=attack, db_url=db_url)


        def create_agent():
            return MockSecGymAgent()
        """
    )
    module_path = tmp_path / "secgym_bootstrap.py"
    module_path.write_text(source, encoding="utf-8")
    return tmp_path, "secgym_bootstrap", "MockSecGymEnv", "MockSecGymAgent"


@pytest.fixture()
def synthesis_project(tmp_path: Path) -> Tuple[Path, str, str, str]:
    source = textwrap.dedent(
        """
        from __future__ import annotations

        from atlas import agent, environment
        from atlas.sdk.interfaces import DiscoveryContext, TelemetryEmitterProtocol


        @environment
        class NeedsConfigEnv:
            def __init__(self, db_url: str, *, cache_dir: str = "/tmp/cache"):
                self._db_url = db_url
                self._cache_dir = cache_dir

            def reset(self, task: str | None = None):
                return {"db": self._db_url, "task": task}

            def step(self, action, submit: bool = False):
                info = {"action": action, "submit": submit, "cache": self._cache_dir}
                return {"status": "pending"}, 0.0, True, info

            def close(self) -> None:
                return None


        @agent
        class NeedsConfigAgent:
            def plan(
                self,
                task: str,
                observation,
                *,
                emit_event: TelemetryEmitterProtocol | None = None,
            ):
                if emit_event:
                    emit_event.emit("analysis", {"task": task})
                return {"goal": "probe connectivity"}

            def act(
                self,
                context: DiscoveryContext,
                *,
                emit_event: TelemetryEmitterProtocol | None = None,
            ):
                if emit_event:
                    emit_event.emit("analysis", {"step": context.step_index})
                return {"ping": True}

            def summarize(
                self,
                context: DiscoveryContext,
                *,
                history=None,
                emit_event: TelemetryEmitterProtocol | None = None,
            ):
                if emit_event:
                    emit_event.emit("analysis", {"summary": True})
                return "Connectivity probe complete"
        """
    )
    module_path = tmp_path / "configurable.py"
    module_path.write_text(source, encoding="utf-8")
    return tmp_path, "configurable", "NeedsConfigEnv", "NeedsConfigAgent"


@pytest.fixture()
def wrapper_only_project(tmp_path: Path) -> Path:
    readme = textwrap.dedent(
        """
        # Wrapper Only Project

        This repo has no Atlas-compatible environment or agent classes.
        Deep integrations live inside the `session` module, which expects `create_runtime`
        to be called with API keys.
        """
    )
    (tmp_path / "README.md").write_text(readme, encoding="utf-8")
    module_source = textwrap.dedent(
        """
        from __future__ import annotations

        import json


        def some_helper_function(config_path: str):
            \"\"\"Some helper that is not an environment/agent factory.\"\"\"
            return {"config": config_path}


        class InternalSession:
            def __init__(self, model: str):
                self._model = model

            def invoke(self, payload):
                return {"model": self._model, "payload": payload}
        """
    )
    (tmp_path / "runtime.py").write_text(module_source, encoding="utf-8")
    return tmp_path
