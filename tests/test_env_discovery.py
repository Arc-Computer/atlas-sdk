from __future__ import annotations

import argparse
import json
from pathlib import Path

from atlas.cli import env as env_cli
from atlas.cli import run as run_cli
from atlas.cli.utils import invoke_discovery_worker
from atlas.sdk.discovery import discover_candidates, split_candidates


def test_discover_candidates_identifies_decorated_classes(stateful_project) -> None:
    project_root, module_name, env_name, agent_name = stateful_project
    candidates = discover_candidates(project_root)
    env_candidates, agent_candidates = split_candidates(candidates)
    assert env_candidates, "expected environment candidates"
    assert agent_candidates, "expected agent candidates"
    assert env_candidates[0].module == module_name
    assert env_candidates[0].qualname == env_name
    assert env_candidates[0].via_decorator is True
    assert agent_candidates[0].qualname == agent_name


def test_discovery_worker_executes_stateful_agent(stateful_project) -> None:
    project_root, module_name, env_name, agent_name = stateful_project
    spec = {
        "project_root": str(project_root),
        "environment": {"module": module_name, "qualname": env_name},
        "agent": {"module": module_name, "qualname": agent_name},
        "task": "Telemetry integration test",
        "run_discovery": True,
        "env": {},
    }
    result = invoke_discovery_worker(spec, timeout=120)
    assert result["final_answer"] == "Completed increments"
    telemetry = result.get("telemetry") or {}
    assert telemetry.get("agent_emitted") is True
    assert telemetry.get("events"), "expected telemetry events to be captured"


def test_env_init_writes_metadata_and_config(stateful_project) -> None:
    project_root, module_name, env_name, agent_name = stateful_project
    args = argparse.Namespace(
        path=str(project_root),
        task="Sample task",
        env_vars=[],
        no_run=False,
        skip_sample_run=True,
        force=True,
        timeout=120,
    )
    exit_code = env_cli._cmd_env_init(args)
    assert exit_code == 0
    atlas_dir = project_root / ".atlas"
    metadata_path = atlas_dir / "discover.json"
    config_path = atlas_dir / "generated_config.yaml"
    assert metadata_path.exists()
    assert config_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["environment"]["module"] == module_name
    assert metadata["agent"]["qualname"] == agent_name
    assert metadata["capabilities"]["control_loop"] == "self"
    assert metadata["telemetry"]["agent_emitted"] is True
    config_text = config_path.read_text(encoding="utf-8")
    assert "behavior: self" in config_text
    assert f"environment: {module_name}:{env_name}" in config_text
    assert f"agent: {module_name}:{agent_name}" in config_text


def test_runtime_rejects_stale_metadata(stateful_project) -> None:
    project_root, module_name, env_name, agent_name = stateful_project
    init_args = argparse.Namespace(
        path=str(project_root),
        task="Sample prompt",
        env_vars=[],
        no_run=True,
        skip_sample_run=True,
        force=True,
        timeout=120,
    )
    assert env_cli._cmd_env_init(init_args) == 0

    run_args = argparse.Namespace(
        path=str(project_root),
        env_vars=[],
        task="Validate run",
        timeout=120,
    )
    first_run_code = run_cli._cmd_run(run_args)
    assert first_run_code == 0
    runs_dir = project_root / ".atlas" / "runs"
    assert any(runs_dir.glob("run_*.json")), "expected run artefact to be created"

    module_path = Path(project_root) / f"{module_name}.py"
    module_path.write_text(module_path.read_text(encoding="utf-8") + "\n# drift", encoding="utf-8")

    stale_exit = run_cli._cmd_run(run_args)
    assert stale_exit == 1
