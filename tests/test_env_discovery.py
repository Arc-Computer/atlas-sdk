from __future__ import annotations

import argparse
import json
from pathlib import Path

from atlas.cli import env as env_cli
from atlas.cli import run as run_cli
from atlas.cli.utils import invoke_discovery_worker
from atlas.sdk.discovery import discover_candidates, split_candidates
from atlas.sdk.factory_synthesis import FactorySnippet, RepositorySummary


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
        env_kwargs=[],
        agent_kwargs=[],
        env_fn=None,
        agent_fn=None,
        env_config=None,
        agent_config=None,
        no_run=False,
        skip_sample_run=True,
        validate=False,
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
    assert metadata["capabilities"]["preferred_mode"] == "auto"
    assert metadata["telemetry"]["agent_emitted"] is True
    assert metadata["environment"]["auto_wrapped"] is False
    assert metadata["agent"]["auto_wrapped"] is False
    config_text = config_path.read_text(encoding="utf-8")
    assert "behavior: self" in config_text
    assert f"environment: {module_name}:{env_name}" in config_text
    assert f"agent: {module_name}:{agent_name}" in config_text
    assert "preferred_mode: auto" in config_text
    assert "forced_mode: auto" in config_text


def test_runtime_rejects_stale_metadata(stateful_project) -> None:
    project_root, module_name, env_name, agent_name = stateful_project
    init_args = argparse.Namespace(
        path=str(project_root),
        task="Sample prompt",
        env_vars=[],
        env_kwargs=[],
        agent_kwargs=[],
        env_fn=None,
        agent_fn=None,
        env_config=None,
        agent_config=None,
        no_run=True,
        skip_sample_run=True,
        validate=False,
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


def test_env_init_persists_discovery_when_database_configured(monkeypatch, stateful_project) -> None:
    project_root, module_name, env_name, agent_name = stateful_project
    captured: list[dict[str, object]] = []

    def fake_persist(**kwargs):
        captured.append(kwargs)
        return 7

    monkeypatch.setenv("STORAGE__DATABASE_URL", "postgresql://stub")
    monkeypatch.setattr(env_cli, "persist_discovery_run", fake_persist)

    args = argparse.Namespace(
        path=str(project_root),
        task="Persist telemetry",
        env_vars=[],
        env_kwargs=[],
        agent_kwargs=[],
        env_fn=None,
        agent_fn=None,
        env_config=None,
        agent_config=None,
        no_run=True,
        skip_sample_run=True,
        validate=False,
        force=True,
        timeout=120,
    )
    exit_code = env_cli._cmd_env_init(args)
    assert exit_code == 0
    assert captured, "expected persistence helper to be invoked"
    entry = captured[0]
    assert entry["task"] == "Persist telemetry"
    assert entry["source"] == "discovery"
    assert entry["project_root"] == project_root
    payload = entry["payload"]
    assert isinstance(payload, dict)
    assert "telemetry" in payload


def test_env_init_auto_skips_heavy_environment(secrl_project) -> None:
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
        validate=False,
        force=True,
        timeout=120,
    )
    exit_code = env_cli._cmd_env_init(args)
    assert exit_code == 0
    metadata_path = project_root / ".atlas" / "discover.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["preflight"]["auto_skip"] is True
    assert metadata["environment"]["factory"]["module"] == module_name
    assert metadata["environment"]["kwargs"]["attack"] == "incident_5"
    assert metadata["agent"]["factory"]["module"] == module_name
    assert metadata["environment"]["auto_wrapped"] is False
    assert metadata["agent"]["auto_wrapped"] is False
    telemetry = metadata["telemetry"]
    assert telemetry.get("events") == []
    assert telemetry.get("agent_emitted") is False
    config_text = (project_root / ".atlas" / "generated_config.yaml").read_text(encoding="utf-8")
    assert "preferred_mode: paired" in config_text
    assert "forced_mode: paired" in config_text


def test_env_init_auto_wraps_when_no_candidates(monkeypatch, wrapper_only_project) -> None:
    project_root = wrapper_only_project
    captured_contexts: list[tuple[str, object]] = []

    def fake_generate(self, role, context, *, previous_error=None):
        captured_contexts.append((role, context))
        if role == "environment":
            return FactorySnippet(
                function_name="auto_environment_factory",
                imports=["import os", "from runtime import build_environment"],
                helpers=[
                    "class AutoWrappedEnvironment:\n"
                    "    def __init__(self, config_path: str, attack: str) -> None:\n"
                    "        self._config = config_path\n"
                    "        self._attack = attack\n"
                    "\n"
                    "    def reset(self, task: str | None = None):\n"
                    "        return {\n"
                    "            \"task\": task,\n"
                    "            \"config\": self._config,\n"
                    "            \"attack\": self._attack,\n"
                    "        }\n"
                    "\n"
                    "    def step(self, action, submit: bool = False):\n"
                    "        return action, 0.0, True, {\"submit\": submit}\n"
                ],
                factory_body=(
                    "def auto_environment_factory(config_path: str = \"configs/runtime.yaml\", **kwargs):\n"
                    "    if os.environ.get('ATLAS_DISCOVERY_VALIDATE') != '1':\n"
                    "        raise RuntimeError('Environment prerequisites not ready; run with --validate once services are up.')\n"
                    "    attack = kwargs.get('attack') or 'incident_9'\n"
                    "    build_environment(config_path=config_path, attack=attack)\n"
                    "    return AutoWrappedEnvironment(config_path=config_path, attack=attack)\n"
                ),
                notes=["Environment wrapper synthesized for validation flow."],
                preflight=["Generated wrapper requires validate mode before execution."],
                auto_skip=True,
            )
        return FactorySnippet(
            function_name="auto_agent_factory",
            imports=["from runtime import build_agent"],
            factory_body=(
                "def auto_agent_factory(model: str = \"gpt-4.1-mini\", **kwargs):\n"
                "    agent = build_agent(model=model)\n"
                "    return agent\n"
            ),
            notes=["Agent wrapper synthesized from repository helpers."],
            preflight=["Review the generated agent wrapper before production use."],
            auto_skip=False,
        )

    monkeypatch.setattr(env_cli.FactorySynthesizer, "_generate_snippet", fake_generate, raising=False)

    args = argparse.Namespace(
        path=str(project_root),
        task="Wrapper bootstrap",
        env_vars=[],
        env_kwargs=["attack=incident_9"],
        agent_kwargs=["model=gpt-4.1-mini"],
        env_fn=None,
        agent_fn=None,
        env_config=None,
        agent_config=None,
        no_run=False,
        skip_sample_run=True,
        validate=False,
        force=True,
        timeout=120,
    )
    exit_code = env_cli._cmd_env_init(args)
    assert exit_code == 0

    assert any(isinstance(ctx, RepositorySummary) for _, ctx in captured_contexts)

    atlas_dir = Path(project_root) / ".atlas"
    factories_source = (atlas_dir / "generated_factories.py").read_text(encoding="utf-8")
    assert "class AutoWrappedEnvironment" in factories_source
    assert "def auto_agent_factory" in factories_source

    metadata = json.loads((atlas_dir / "discover.json").read_text(encoding="utf-8"))
    assert metadata["environment"]["auto_wrapped"] is True
    assert metadata["agent"]["auto_wrapped"] is True
    assert metadata["preflight"]["auto_skip"] is True
    assert any("wrapper" in note.lower() for note in metadata["preflight"]["notes"])

    config_text = (atlas_dir / "generated_config.yaml").read_text(encoding="utf-8")
    assert "forced_mode: paired" in config_text


def test_env_init_synthesizes_factories_with_auto_skip(monkeypatch, synthesis_project) -> None:
    project_root, module_name, env_name, agent_name = synthesis_project

    env_snippet = FactorySnippet(
        function_name="create_environment_factory",
        imports=[
            "import os",
            f"from {module_name} import {env_name}",
        ],
        helpers=[],
        factory_body=(
            "def create_environment_factory(*, db_url: str | None = None, **kwargs):\n"
            "    if os.environ.get('ATLAS_DISCOVERY_VALIDATE') != '1':\n"
            "        raise RuntimeError('Database not ready for validation')\n"
            "    defaults = {'db_url': db_url or 'sqlite:///memory.db'}\n"
            "    defaults.update(kwargs)\n"
            f"    return {env_name}(**defaults)\n"
        ),
        notes=["LLM: ensured sqlite fallback"],
        preflight=["Start the database container before validation."],
        auto_skip=True,
    )
    agent_snippet = FactorySnippet(
        function_name="create_agent_factory",
        imports=[f"from {module_name} import {agent_name}"],
        helpers=[],
        factory_body=(
            "def create_agent_factory(**kwargs):\n"
            f"    return {agent_name}()\n"
        ),
        notes=[],
        preflight=[],
        auto_skip=False,
    )

    def fake_generate(self, role, context, *, previous_error=None):
        return env_snippet if role == "environment" else agent_snippet

    monkeypatch.setattr(env_cli.FactorySynthesizer, "_generate_snippet", fake_generate, raising=False)

    args = argparse.Namespace(
        path=str(project_root),
        task="Investigate connectivity",
        env_vars=[],
        env_kwargs=[],
        agent_kwargs=[],
        env_fn=None,
        agent_fn=None,
        env_config=None,
        agent_config=None,
        no_run=False,
        skip_sample_run=True,
        validate=False,
        force=True,
        timeout=120,
    )
    exit_code = env_cli._cmd_env_init(args)
    assert exit_code == 0

    atlas_dir = project_root / ".atlas"
    factories_path = atlas_dir / "generated_factories.py"
    assert factories_path.exists()
    factory_source = factories_path.read_text(encoding="utf-8")
    assert "def create_environment_factory" in factory_source

    metadata_path = atlas_dir / "discover.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["preflight"]["auto_skip"] is True
    notes = metadata["preflight"]["notes"]
    assert any("database container" in note.lower() for note in notes)
    assert metadata["environment"]["factory"]["module"] == ".atlas.generated_factories"
    assert metadata["capabilities"]["preferred_mode"] == "paired"
    assert metadata["synthesis"]["notes"] == env_snippet.notes
    assert "factory" not in metadata["agent"]
    assert metadata["environment"]["auto_wrapped"] is False
    assert metadata["agent"]["auto_wrapped"] is False

    config_text = (atlas_dir / "generated_config.yaml").read_text(encoding="utf-8")
    assert "preferred_mode: paired" in config_text
    assert "forced_mode: paired" in config_text


def test_env_scaffold_writes_template(tmp_path: Path) -> None:
    args = argparse.Namespace(
        template="langgraph",
        output=None,
        path=str(tmp_path),
        force=False,
    )
    exit_code = env_cli._cmd_env_scaffold(args)
    assert exit_code == 0
    generated = tmp_path / "langgraph_adapter.py"
    assert generated.exists()
    content = generated.read_text(encoding="utf-8")
    assert "LangGraphEnvironment" in content
    assert "create_agent" in content
