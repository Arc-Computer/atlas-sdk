from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from atlas.cli import env as env_cli


def test_validation_always_runs_regardless_of_auto_skip(stateful_project, capsys: pytest.CaptureFixture[str]) -> None:
    project_root, module_name, env_name, agent_name = stateful_project
    args = argparse.Namespace(
        path=str(project_root),
        task="Sample task",
        env_vars=[],
        env_kwargs=[],
        agent_kwargs=[],
        env_fn=None,
        agent_fn=None,
        env_config=None,
        agent_config=None,
        no_run=False,
        skip_sample_run=True,
        scaffold_config_full=False,
        force=True,
        timeout=120,
    )
    exit_code = env_cli._cmd_env_init(args)
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Validation succeeded" in captured.out or "Validation failed" in captured.err
    assert "Validation deferred" not in captured.out
    assert "Validation deferred" not in captured.err
    assert "--validate" not in captured.out
    assert "--validate" not in captured.err
    marker_path = project_root / ".atlas" / ".validated"
    assert marker_path.exists()


def test_validation_marker_written_on_success(stateful_project) -> None:
    project_root, module_name, env_name, agent_name = stateful_project
    args = argparse.Namespace(
        path=str(project_root),
        task="Sample task",
        env_vars=[],
        env_kwargs=[],
        agent_kwargs=[],
        env_fn=None,
        agent_fn=None,
        env_config=None,
        agent_config=None,
        no_run=False,
        skip_sample_run=True,
        scaffold_config_full=False,
        force=True,
        timeout=120,
    )
    exit_code = env_cli._cmd_env_init(args)
    assert exit_code == 0
    marker_path = project_root / ".atlas" / ".validated"
    assert marker_path.exists()
    marker_data = json.loads(marker_path.read_text(encoding="utf-8"))
    assert "validated_at" in marker_data


def test_validation_marker_written_on_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    project_root = tmp_path / "broken_project"
    project_root.mkdir()
    broken_module = project_root / "broken.py"
    broken_module.write_text(
        "class BrokenEnv:\n    def __init__(self, missing_required):\n        pass\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        path=str(project_root),
        task="Sample task",
        env_vars=[],
        env_kwargs=[],
        agent_kwargs=[],
        env_fn="broken:BrokenEnv",
        agent_fn=None,
        env_config=None,
        agent_config=None,
        no_run=False,
        skip_sample_run=True,
        scaffold_config_full=False,
        force=True,
        timeout=120,
    )
    exit_code = env_cli._cmd_env_init(args)
    marker_path = project_root / ".atlas" / ".validated"
    assert marker_path.exists()
    marker_data = json.loads(marker_path.read_text(encoding="utf-8"))
    assert "validated_at" in marker_data


def test_error_message_no_validate_reference(stateful_project) -> None:
    project_root, module_name, env_name, agent_name = stateful_project
    args = argparse.Namespace(
        path=str(project_root),
        task="Sample task",
        env_vars=[],
        env_kwargs=[],
        agent_kwargs=[],
        env_fn=None,
        agent_fn=None,
        env_config=None,
        agent_config=None,
        no_run=False,
        skip_sample_run=True,
        scaffold_config_full=False,
        force=True,
        timeout=120,
    )
    env_cli._cmd_env_init(args)
    generated_factories = project_root / ".atlas" / "generated_factories.py"
    if generated_factories.exists():
        factory_content = generated_factories.read_text(encoding="utf-8")
        assert "_atlas_require_validation" in factory_content
        assert "atlas env init --validate" not in factory_content
        assert "atlas env init" in factory_content or "Run 'atlas env init'" in factory_content


def test_validation_output_with_auto_skip(secrl_project, capsys: pytest.CaptureFixture[str]) -> None:
    project_root, module_name, env_name, agent_name = secrl_project
    args = argparse.Namespace(
        path=str(project_root),
        task="Investigate attack",
        env_vars=[],
        env_kwargs=["attack=incident_5", "db_url=mysql://root@localhost"],
        agent_kwargs=[],
        env_fn=f"{module_name}:create_environment",
        agent_fn=f"{module_name}:create_agent",
        env_config=None,
        agent_config=None,
        no_run=False,
        skip_sample_run=True,
        scaffold_config_full=False,
        force=True,
        timeout=120,
    )
    exit_code = env_cli._cmd_env_init(args)
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Validation succeeded" in captured.out or "Validation failed" in captured.err
    assert "Validation deferred" not in captured.out
    assert "Validation deferred" not in captured.err
    marker_path = project_root / ".atlas" / ".validated"
    assert marker_path.exists()

