import os
import subprocess
import sys
import textwrap
from pathlib import Path

import yaml


def _write_demo_project(project_root: Path) -> None:
    (project_root / "demo").mkdir(parents=True, exist_ok=True)
    (project_root / "demo" / "__init__.py").write_text("", encoding="utf-8")
    (project_root / ".env").write_text("DEMO_FROM_ENV=1\n", encoding="utf-8")
    factories = textwrap.dedent(
        """\
        from __future__ import annotations

        from typing import Any, Dict

        from atlas.sdk.interfaces import AtlasAgentProtocol, AtlasEnvironmentProtocol, DiscoveryContext, TelemetryEmitterProtocol


        class DemoEnvironment(AtlasEnvironmentProtocol):
            def __init__(self) -> None:
                self._step = 0

            def reset(self, task: str | None = None) -> Dict[str, Any]:
                self._step = 0
                return {"task": task or "demo-task", "steps": []}

            def step(self, action: Dict[str, Any] | Any, submit: bool = False):
                self._step += 1
                observation = {"steps": self._step}
                done = submit or self._step >= 2
                reward = 1.0 if done else 0.0
                info = {"submit": submit}
                return observation, reward, done, info

            def close(self) -> None:
                return None


        class DemoAgent(AtlasAgentProtocol):
            def __init__(self) -> None:
                self._last_action = None

            def plan(
                self,
                task: str,
                observation: Any,
                *,
                emit_event: TelemetryEmitterProtocol | None = None,
            ) -> Dict[str, Any]:
                if emit_event:
                    emit_event.emit("progress", {"stage": "plan"})
                return {"steps": [{"id": 1, "description": task, "depends_on": []}]}

            def act(
                self,
                context: DiscoveryContext,
                *,
                emit_event: TelemetryEmitterProtocol | None = None,
            ) -> Dict[str, Any]:
                submit = context.step_index >= 1
                action = {"message": f"demo-step-{context.step_index}"}
                if emit_event:
                    emit_event.emit("progress", {"stage": "act", "submit": submit})
                self._last_action = action
                return {"action": action, "submit": submit}

            def summarize(
                self,
                context: DiscoveryContext,
                *,
                history=None,
                emit_event: TelemetryEmitterProtocol | None = None,
            ) -> str:
                return "Demo summary"


        def create_environment(**_: Any) -> DemoEnvironment:
            return DemoEnvironment()


        def create_agent(prompt: str | None = None, metadata: Dict[str, Any] | None = None, **_: Any):
            if prompt is not None:
                return "Demo agent response"
            return DemoAgent()
        """
    )
    (project_root / "demo" / "factories.py").write_text(factories, encoding="utf-8")


def _run_cli(args: list[str], *, cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [sys.executable, "-m", "atlas.cli.main", *args],
        cwd=str(cwd),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(f"Command failed: {' '.join(args)}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")
    return completed


def test_env_init_scaffolds_runnable_config(tmp_path: Path) -> None:
    project_root = tmp_path / "demo_project"
    project_root.mkdir()
    _write_demo_project(project_root)

    repo_root = Path(__file__).resolve().parents[2]
    base_env = os.environ.copy()
    base_env.setdefault("ATLAS_FAKE_LLM", "1")

    _run_cli(
        [
            "env",
            "init",
            "--path",
            str(project_root),
            "--env-fn",
            "demo.factories:create_environment",
            "--agent-fn",
            "demo.factories:create_agent",
            "--scaffold-config-full",
            "--force",
            "--no-run",
        ],
        cwd=repo_root,
        env=base_env,
    )

    generated_config = project_root / ".atlas" / "generated_config.yaml"
    assert generated_config.exists()

    rendered = yaml.safe_load(generated_config.read_text(encoding="utf-8"))
    assert rendered["agent"]["type"] == "python"
    assert rendered["agent"]["import_path"] == "demo.factories"
    assert "response_format" not in rendered["agent"]

    _run_cli(
        [
            "run",
            "--path",
            str(project_root),
            "--config",
            str(generated_config),
            "--task",
            "Smoke test",
        ],
        cwd=repo_root,
        env=base_env,
    )
