"""Environment onboarding CLI commands."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import shlex
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

try:
    import yaml
except Exception:
    yaml = None

from atlas.cli.persistence import persist_discovery_run
from atlas.cli.utils import (
    CLIError,
    DiscoveryWorkerError,
    invoke_discovery_worker,
    load_config_file,
    parse_callable_reference,
    parse_env_flags,
    parse_key_value_flags,
)
from atlas.config.models import LearningConfig, RuntimeSafetyConfig
from atlas.sdk.discovery import (
    Candidate,
    Role,
    collect_runtime_metadata,
    discover_candidates,
    serialize_candidate,
    split_candidates,
    write_discovery_payload,
)
from atlas.sdk.llm_inference import (
    LLM_PROVIDER_DEFAULT_ENV,
    collect_llm_candidate_entries,
    merge_llm_block,
    select_llm_candidate,
)
from atlas.cli.env_types import (
    DISCOVERY_FILENAME,
    GENERATED_CONFIG_FILENAME,
    VALIDATION_MARKER_FILENAME,
    SelectedTargets,
    TargetSpec,
)
from atlas.sdk.factory_synthesis import (
    AGENT_FUNCTION_NAME,
    AGENT_RUNTIME_HELPER,
    ENV_FUNCTION_NAME,
    ENV_VALIDATE_FLAG,
    GENERATED_MODULE,
    FactorySnippet,
    FactorySynthesizer,
    build_agent_factory_snippet,
    build_environment_factory_snippet,
)
SCAFFOLD_TEMPLATES = {
    "langgraph": {
        "filename": "langgraph_adapter.py",
        "content": textwrap.dedent(
            """\
            \"\"\"Atlas factory helpers for LangGraph-style agents.\"\"\"

            from __future__ import annotations

            from typing import Any, Dict

            from atlas.sdk.interfaces import DiscoveryContext, TelemetryEmitterProtocol


            class LangGraphEnvironment:
                \"\"\"Minimal environment shim you can customise for your stack.\"\"\"

                def __init__(self, *, dataset: str = \"incidents\" ) -> None:
                    self._dataset = dataset
                    self._step = 0

                def reset(self, task: str | None = None) -> Dict[str, Any]:
                    self._step = 0
                    return {\"task\": task or \"Investigate incident\", \"dataset\": self._dataset}

                def step(self, action: Dict[str, Any], submit: bool = False):
                    self._step += 1
                    done = submit or self._step >= 3
                    observation = {
                        \"step\": self._step,
                        \"action\": action,
                        \"submit\": submit,
                    }
                    reward = 1.0 if done else 0.0
                    info = {\"dataset\": self._dataset}
                    return observation, reward, done, info

                def close(self) -> None:  # pragma: no cover - placeholder cleanup
                    return None


            class LangGraphAgentWrapper:
                \"\"\"Wrap a compiled LangGraph graph so Atlas can orchestrate it.\"\"\"

                def __init__(self, graph) -> None:
                    self._graph = graph
                    self._state: Dict[str, Any] = {}

                def plan(
                    self,
                    task: str,
                    observation: Any,
                    *,
                    emit_event: TelemetryEmitterProtocol | None = None,
                ) -> Dict[str, Any]:
                    self._state = {\"task\": task, \"observation\": observation}
                    if emit_event:
                        emit_event.emit(\"progress\", {\"stage\": \"plan\", \"task\": task})
                    return {\"execution\": \"graph\", \"note\": \"LangGraph adapter\"}

                def act(
                    self,
                    context: DiscoveryContext,
                    *,
                    emit_event: TelemetryEmitterProtocol | None = None,
                ) -> Dict[str, Any]:
                    if emit_event:
                        emit_event.emit(\"progress\", {\"stage\": \"act\", \"step\": context.step_index})
                    result = self._graph.invoke({\"observation\": context.observation, \"state\": self._state})
                    action = result.get(\"action\") if isinstance(result, dict) else result
                    submit = bool(result.get(\"submit\")) if isinstance(result, dict) else False
                    return {\"action\": action, \"submit\": submit}

                def summarize(
                    self,
                    context: DiscoveryContext,
                    *,
                    history=None,
                    emit_event: TelemetryEmitterProtocol | None = None,
                ) -> str:
                    if emit_event:
                        emit_event.emit(\"progress\", {\"stage\": \"summarize\"})
                    return str(context.observation)


            def create_environment(dataset: str = \"incidents\") -> LangGraphEnvironment:
                \"\"\"Factory passed via --env-fn in `atlas env init`.\"\"\"

                return LangGraphEnvironment(dataset=dataset)


            def create_agent(graph) -> LangGraphAgentWrapper:
                \"\"\"Factory passed via --agent-fn in `atlas env init`.\"\"\"

                return LangGraphAgentWrapper(graph)
            """
        ),
    }
}



FULL_CONFIG_TEMPLATE = "openai_agent.yaml"


def _validate_discovered_artifacts(
    project_root: Path,
    atlas_dir: Path,
    environment: TargetSpec,
    agent: TargetSpec,
) -> tuple[bool, list[str]]:
    generated_path = atlas_dir / "generated_factories.py"
    if not generated_path.exists():
        return False, [f"Generated factories not found at {generated_path}"]

    module_name = f"{GENERATED_MODULE}_validation"
    spec = importlib.util.spec_from_file_location(module_name, generated_path)
    if spec is None or spec.loader is None:
        return False, ["Unable to load generated factories module"]

    added_path = None
    project_str = str(project_root)
    if project_str not in sys.path:
        sys.path.insert(0, project_str)
        added_path = project_str

    original_skip = os.environ.get("ATLAS_SKIP_VALIDATION")
    os.environ["ATLAS_SKIP_VALIDATION"] = "1"

    errors: list[str] = []
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        def _validate(target: TargetSpec, kind: str) -> None:
            if target.factory is None:
                return
            module_ref, qualname = target.factory
            if module_ref != GENERATED_MODULE:
                return
            func = getattr(module, qualname, None)
            if not callable(func):
                errors.append(f"{kind.title()} factory {qualname} not found in generated module")
                return
            kwargs = target.kwargs if isinstance(target.kwargs, dict) else {}
            try:
                instance = func(**kwargs)
                close = getattr(instance, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
            except Exception as exc:  # pragma: no cover - validation safety net
                errors.append(f"{kind.title()} validation failed: {exc}")

        _validate(environment, "environment")
        _validate(agent, "agent")
    except Exception as exc:  # pragma: no cover - defensive
        errors.append(f"Validation failed: {exc}")
    finally:
        if original_skip is None:
            os.environ.pop("ATLAS_SKIP_VALIDATION", None)
        else:
            os.environ["ATLAS_SKIP_VALIDATION"] = original_skip
        sys.modules.pop(module_name, None)
        if added_path is not None:
            try:
                sys.path.remove(added_path)
            except ValueError:
                pass

    return (not errors), errors


def _load_project_env(project_root: Path) -> dict[str, str]:
    """Load environment variables from a .env file without requiring python-dotenv."""

    env_path = project_root / ".env"
    if not env_path.exists():
        return {}
    try:
        raw = env_path.read_text(encoding="utf-8")
    except OSError:
        return {}
    result: dict[str, str] = {}
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.lower().startswith("export "):
            stripped = stripped[7:].strip()
        key, sep, value = stripped.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        result.setdefault(key, value)
    return result


def _extract_module_from_string(value: object) -> Sequence[str]:
    if not isinstance(value, str):
        return []
    if ":" in value:
        module, _, _ = value.partition(":")
        return [module.strip()] if module.strip() else []
    if "." in value and "/" not in value and " " not in value:
        return [value.strip()]
    return []


def _collect_candidate_modules(target: TargetSpec) -> set[str]:
    modules: set[str] = set()
    module, _ = _resolve_callable_reference(target)
    if module:
        modules.add(module)
    for maybe_module in target.kwargs.values():
        modules.update(_extract_module_from_string(maybe_module))
    if target.config:
        for maybe_module in target.config.values():
            modules.update(_extract_module_from_string(maybe_module))
    return modules


def _infer_pythonpath_entries(project_root: Path, module_path: str) -> list[str]:
    """Return directories that should be added to PYTHONPATH for a given module path."""

    if not module_path or module_path.startswith("."):
        return []
    path_parts = [part for part in module_path.split(".") if part]
    if not path_parts:
        return []
    base_path = project_root.joinpath(*path_parts)
    file_candidate = base_path.with_suffix(".py")
    if file_candidate.exists():
        target_path = file_candidate
    elif base_path.exists():
        target_path = base_path
    else:
        return []
    entries: list[str] = []
    for parent in [target_path.parent, *target_path.parents]:
        if not parent or not parent.exists():
            continue
        entries.append(str(parent))
        if parent == project_root:
            break
    return entries


def _prepare_pythonpath_overrides(project_root: Path, targets: SelectedTargets) -> list[str]:
    modules: set[str] = set()
    modules.update(_collect_candidate_modules(targets.environment))
    modules.update(_collect_candidate_modules(targets.agent))
    entries: list[str] = []
    seen: set[str] = set()
    for module in modules:
        for entry in _infer_pythonpath_entries(project_root, module):
            if entry not in seen:
                seen.add(entry)
                entries.append(entry)
    return entries


def _serialize_target(target: TargetSpec, project_root: Path, role: Role) -> dict[str, object]:
    payload: dict[str, object] = {
        "selection": "factory" if target.candidate is None else "candidate",
        "kwargs": target.kwargs,
        "config": target.config,
        "role": role,
    }
    if target.candidate is not None:
        payload.update(serialize_candidate(target.candidate, project_root))
        payload["capabilities"] = target.candidate.capabilities
    if target.factory is not None:
        payload.setdefault("module", target.factory[0])
        payload.setdefault("qualname", target.factory[1])
        payload["factory"] = {
            "module": target.factory[0],
            "qualname": target.factory[1],
        }
    payload["auto_wrapped"] = target.auto_wrapped
    # Ensure required keys exist even if no candidate was discovered.
    payload.setdefault("module", None)
    payload.setdefault("qualname", None)
    payload.setdefault("file", None)
    payload.setdefault("hash", None)
    return payload


def _filter_candidates(candidates: List[Candidate]) -> List[Candidate]:
    if len(candidates) <= 1:
        return candidates
    filtered = [
        candidate
        for candidate in candidates
        if not (
            candidate.reason == "heuristic"
            and candidate.module.startswith("atlas.sdk.")
        )
    ]
    return filtered or candidates


def _auto_select_candidate(
    candidates: List[Candidate],
    preferred_module: str | None,
) -> tuple[Candidate | None, bool]:
    if not candidates:
        return None, False
    filtered = _filter_candidates(candidates)
    if preferred_module:
        for candidate in filtered:
            if candidate.module == preferred_module:
                return candidate, True
    if len(filtered) == 1:
        return filtered[0], True
    decorated = [candidate for candidate in filtered if candidate.via_decorator]
    if len(decorated) == 1:
        return decorated[0], True
    sorted_candidates = sorted(
        filtered,
        key=lambda candidate: (candidate.via_decorator, candidate.score),
        reverse=True,
    )
    return sorted_candidates[0], False


def _summarise_failure_hints(exc: DiscoveryWorkerError) -> list[str]:
    hints: list[str] = []
    trace = exc.traceback or ""
    message = str(exc)
    payload = f"{trace}\n{message}".lower()
    if "modulenotfounderror" in payload:
        hints.append(
            "Module import failed inside discovery. Ensure project dependencies are installed and PYTHONPATH points to your source tree (define PYTHONPATH in .env if needed)."
        )
    if "validationerror" in payload or "pydantic_core" in payload:
        hints.append(
            "Factory initialisation raised validation errors. Double-check required environment variables (atlas env init now auto-loads .env but secrets may still be missing)."
        )
    if "keyerror: 'data'" in payload:
        hints.append(
            "LangChain tool schema reported missing field 'data'. Update your StructuredTool args_schema to include that field or upgrade the auth0-ai tool wrappers."
        )
    return hints


def _filter_actionable_prerequisites(skip_reasons: list[str]) -> list[str]:
    """Extract only actionable prerequisites, filtering out internal notes.

    TODO: This is a band-aid. The proper fix is to separate internal_notes from
    prerequisites at the source (FactorySnippet, synthesis). See refactoring issue.
    """
    actionable = []

    # Internal phrases to filter out
    internal_phrases = [
        "instantiated via factory only",
        "skipping automatic run",
        "review generated factory",
        "synthesized automatically",
        "validation is required",
        "deferred until",
        "atlas_discovery_validate",
        "enable environment validation",
        "enable validation",
        "module is importable in the",
        "discovery environment",
    ]

    for reason in skip_reasons:
        reason_lower = reason.lower()

        # Skip internal implementation notes
        if any(phrase in reason_lower for phrase in internal_phrases):
            continue

        # Keep actionable items (install, config, etc)
        if any(keyword in reason_lower for keyword in [
            "install", "pip install", "dependencies",
            "ensure", "set ", "configure", "api key",
            "missing", "required"
        ]):
            actionable.append(reason.strip())

    return actionable


def _print_factory_hint(role: str) -> None:
    print(f"\u2717 No {role} found", file=sys.stderr)
    print(file=sys.stderr)
    print(f"To specify {role} manually:", file=sys.stderr)
    role_flag = "env" if role == "environment" else "agent"
    print(f"  atlas env init --{role_flag}-fn your.module:name", file=sys.stderr)
    print(file=sys.stderr)
    print("Documentation: https://docs.arc.computer/sdk/agent-patterns", file=sys.stderr)


def _cmd_env_scaffold(args: argparse.Namespace) -> int:
    template_key = (args.template or "langgraph").lower()
    template = SCAFFOLD_TEMPLATES.get(template_key)
    if template is None:
        print(
            f"Unsupported template {args.template!r}. Available options: {', '.join(sorted(SCAFFOLD_TEMPLATES))}.",
            file=sys.stderr,
        )
        return 1
    project_root = Path(args.path or ".").resolve()
    destination = project_root / (args.output or template["filename"])
    if destination.exists() and not args.force:
        print(f"{destination} already exists. Use --force to overwrite.", file=sys.stderr)
        return 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(template["content"], encoding="utf-8")
    print(f"Scaffolded {template_key!r} factory helpers at {destination}")
    print("Update the template to call your own constructors and supply real LangGraph graphs.")
    return 0


def _prompt_selection(candidates: List[Candidate], role: str) -> Candidate:
    if not candidates:
        raise ValueError(f"No candidates detected for role '{role}'.")
    if len(candidates) == 1:
        return candidates[0]
    print(f"Multiple {role} candidates detected:")
    for index, candidate in enumerate(candidates, start=1):
        marker = "[decorator]" if candidate.via_decorator else "[heuristic]"
        capability_summary = _summarise_capabilities(candidate)
        print(f"  {index}. {candidate.dotted_path()} {marker} score={candidate.score} {capability_summary}")
    while True:
        try:
            raw = input(f"Select {role} [1-{len(candidates)}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nDefaulting to first candidate.")
            return candidates[0]
        if not raw:
            print("Please provide a selection.")
            continue
        if not raw.isdigit():
            print("Selection must be a number.")
            continue
        choice = int(raw)
        if 1 <= choice <= len(candidates):
            return candidates[choice - 1]
        print(f"Selection out of range. Choose between 1 and {len(candidates)}.")


def _summarise_capabilities(candidate: Candidate) -> str:
    caps = candidate.capabilities or {}
    if candidate.role == "environment":
        ordered = [("reset", "R"), ("step", "S"), ("close", "C")]
    else:
        ordered = [("plan", "P"), ("act", "A"), ("summarize", "Z")]
    flags = "".join(label if caps.get(key) else label.lower() for key, label in ordered)
    return f"(caps:{flags})"


def _ensure_write(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists. Use --force to overwrite.")


def _summarise_target(target: TargetSpec, role: str) -> str:
    parts: list[str] = []
    if target.candidate is not None:
        source = "decorator" if target.candidate.via_decorator else target.candidate.reason
        parts.append(f"candidate={target.candidate.dotted_path()} ({source})")
        parts.append(_summarise_capabilities(target.candidate))
    if target.factory is not None:
        parts.append(f"factory={target.factory[0]}:{target.factory[1]}")
    if target.kwargs:
        parts.append(f"kwargs={len(target.kwargs)}")
    if target.auto_wrapped:
        parts.append("auto_wrapped")
    return f"{role}: " + ", ".join(parts or ["<unspecified>"])


def _infer_skip_reasons(targets: SelectedTargets) -> list[str]:
    reasons: list[str] = []
    env_path = targets.environment.dotted_path().lower()
    agent_path = targets.agent.dotted_path().lower()
    if "secgym" in env_path or "mysql" in env_path:
        reasons.append("Environment references SecGym/mysql; ensure Docker containers and database are running before executing discovery.")
    if "deepagents" in agent_path or "langgraph" in agent_path:
        reasons.append("LangGraph/DeepAgents detected; ensure dependencies (uv/poetry) are installed before executing discovery.")
    if targets.environment.factory and not targets.environment.candidate:
        reasons.append("Environment instantiated via factory only; skipping automatic run until validated.")
    if targets.agent.candidate and not targets.agent.candidate.capabilities.get("summarize"):
        reasons.append("Agent missing summarize(); Atlas wrapper will auto-generate final answers. Validate behaviour manually before running discovery.")
    return reasons


def _resolve_callable_reference(target: TargetSpec) -> tuple[str | None, str | None]:
    if target.factory is not None:
        return target.factory
    if target.candidate is not None:
        return target.candidate.module, target.candidate.qualname
    return None, None


def _build_factory_metadata(target: TargetSpec) -> dict[str, Any] | None:
    module, qualname = _resolve_callable_reference(target)
    if not module or not qualname:
        return None
    payload: dict[str, Any] = {
        "module": module,
        "qualname": qualname,
        "working_directory": "./",
    }
    if target.kwargs:
        payload["kwargs"] = dict(target.kwargs)
    if target.config:
        payload["config"] = target.config
    if target.auto_wrapped:
        payload["auto_wrapped"] = True
    return payload


def _build_llm_block(template_block: dict[str, Any] | None, provider: str | None, model: str | None) -> dict[str, Any]:
    block: dict[str, Any] = copy.deepcopy(template_block) if isinstance(template_block, dict) else {}
    original_provider = block.get("provider") if isinstance(block.get("provider"), str) else None
    if provider:
        block["provider"] = provider
    else:
        provider = original_provider
    if model:
        block["model"] = model
    if provider:
        default_env = LLM_PROVIDER_DEFAULT_ENV.get(provider)
        api_env = block.get("api_key_env")
        if default_env and (not api_env or original_provider != provider or api_env == LLM_PROVIDER_DEFAULT_ENV.get(original_provider or "")):
            block["api_key_env"] = default_env
    return block


def _load_full_config_template() -> tuple[dict[str, Any] | None, dict[str, Any]]:
    info: dict[str, Any] = {}
    if yaml is None:
        info["reason"] = "pyyaml-missing"
        return None, info

    raw: str | None = None

    try:
        from importlib import resources as importlib_resources

        template_resource = importlib_resources.files("atlas.templates").joinpath(FULL_CONFIG_TEMPLATE)
        if template_resource.is_file():
            raw = template_resource.read_text(encoding="utf-8")
            info["template_path"] = str(template_resource)
            info["template_source"] = "package"
    except (ModuleNotFoundError, FileNotFoundError, AttributeError):
        template_resource = None

    if raw is None:
        template_root = Path(__file__).resolve().parents[2] / "configs" / "examples"
        template_path = template_root / FULL_CONFIG_TEMPLATE
        info.setdefault("template_path", str(template_path))
        if not template_path.exists():
            info["reason"] = "template-missing"
            return None, info
        try:
            raw = template_path.read_text(encoding="utf-8")
        except OSError as exc:
            info["reason"] = f"template-read-error: {exc}"
            return None, info

    try:
        payload = yaml.safe_load(raw) or {}
    except Exception as exc:
        info["reason"] = f"template-parse-error: {exc}"
        return None, info
    if not isinstance(payload, dict):
        info["reason"] = "template-invalid"
        return None, info
    return payload, info


def _compose_full_config_payload(
    template_payload: dict[str, Any],
    targets: SelectedTargets,
    project_root: Path,
    llm_capabilities: dict[str, Any],
    capabilities: dict[str, object] | None = None,
    runtime_metadata: dict[str, dict[str, object]] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    capabilities = capabilities or {}
    runtime_metadata = runtime_metadata or {}
    config_payload = copy.deepcopy(template_payload)
    config_payload.setdefault("learning", {})
    config_payload.setdefault("runtime_safety", {})
    agent_module, agent_qualname = _resolve_callable_reference(targets.agent)
    if not agent_module or not agent_qualname:
        return None, {"reason": "agent-missing"}
    agent_template = config_payload.get("agent") if isinstance(config_payload.get("agent"), dict) else {}
    agent_block: dict[str, Any] = {
        "type": "python",
        "import_path": agent_module,
        "attribute": agent_qualname,
        "working_directory": "./",
        "allow_generator": False,
    }
    template_name = agent_template.get("name") if isinstance(agent_template, dict) else None
    default_agent_name = template_name if isinstance(template_name, str) and template_name.strip() else f"{project_root.name}-agent"
    agent_block["name"] = default_agent_name
    system_prompt = agent_template.get("system_prompt") if isinstance(agent_template, dict) else ""
    if isinstance(system_prompt, str):
        agent_block["system_prompt"] = system_prompt
    tools_section = agent_template.get("tools") if isinstance(agent_template, dict) else []
    if isinstance(tools_section, list):
        agent_block["tools"] = copy.deepcopy(tools_section)
    agent_runtime_meta = runtime_metadata.get("agent") if isinstance(runtime_metadata, dict) else {}
    prompt_literals = agent_runtime_meta.get("prompts") if isinstance(agent_runtime_meta, dict) else []
    if isinstance(prompt_literals, list) and prompt_literals:
        agent_block["system_prompt"] = prompt_literals[0]
    tool_entries = agent_runtime_meta.get("tools") if isinstance(agent_runtime_meta, dict) else []
    if isinstance(tool_entries, list) and tool_entries:
        normalized_tools: list[dict[str, Any]] = []
        for entry in tool_entries:
            if isinstance(entry, dict):
                tool_payload = dict(entry)
                name = tool_payload.get("name") or tool_payload.get("id")
                if not name:
                    continue
                tool_payload["name"] = str(name)
                tool_payload.setdefault("description", "Provided by repository metadata.")
                normalized_tools.append({"name": tool_payload["name"], "description": tool_payload["description"]})
            elif isinstance(entry, str):
                normalized_tools.append({"name": entry, "description": "Provided by repository metadata."})
            else:
                normalized_tools.append({"name": str(entry), "description": "Provided by repository metadata."})
        if normalized_tools:
            agent_block["tools"] = normalized_tools

    llm_provider = llm_capabilities.get("provider") if isinstance(llm_capabilities, dict) else None
    llm_model = llm_capabilities.get("model") if isinstance(llm_capabilities, dict) else None
    llm_source = llm_capabilities.get("source") if isinstance(llm_capabilities, dict) else None
    llm_template = agent_template.get("llm") if isinstance(agent_template, dict) else None
    llm_block = _build_llm_block(llm_template, llm_provider, llm_model)
    llm_candidates = collect_llm_candidate_entries(agent_runtime_meta if isinstance(agent_runtime_meta, dict) else None)
    selected_llm_candidate = select_llm_candidate(llm_candidates)
    if selected_llm_candidate:
        llm_block = merge_llm_block(llm_block, selected_llm_candidate)
        provider_override = selected_llm_candidate.get("provider")
        model_override = selected_llm_candidate.get("model")
        if isinstance(provider_override, str) and provider_override:
            llm_provider = provider_override
        if isinstance(model_override, str) and model_override:
            llm_model = model_override
        if not llm_source:
            llm_source = selected_llm_candidate.get("source") or "repository"
    if llm_block:
        agent_block["llm"] = llm_block
        llm_provider = llm_block.get("provider", llm_provider)
        llm_model = llm_block.get("model", llm_model)
    config_payload["agent"] = agent_block

    teacher_template = config_payload.get("teacher") if isinstance(config_payload.get("teacher"), dict) else {}
    teacher_block = copy.deepcopy(teacher_template)
    teacher_block["llm"] = _build_llm_block(teacher_template.get("llm") if isinstance(teacher_template, dict) else None, llm_provider, llm_model)
    config_payload["teacher"] = teacher_block

    metadata_block_current = config_payload.get("metadata") if isinstance(config_payload.get("metadata"), dict) else {}
    discovery_meta: dict[str, Any] = {}
    env_runtime_meta = runtime_metadata.get("environment") if isinstance(runtime_metadata, dict) else {}
    if isinstance(env_runtime_meta, dict) and env_runtime_meta.get("parameters"):
        discovery_meta["environment_parameters"] = env_runtime_meta["parameters"]
    if isinstance(agent_runtime_meta, dict):
        if agent_runtime_meta.get("parameters"):
            discovery_meta["agent_parameters"] = agent_runtime_meta["parameters"]
        if agent_runtime_meta.get("factory_kwargs"):
            discovery_meta["agent_factory_kwargs"] = agent_runtime_meta["factory_kwargs"]
        if agent_block.get("system_prompt"):
            discovery_meta["agent_prompt_preview"] = agent_block["system_prompt"][:2000]
    env_metadata = _build_factory_metadata(targets.environment)
    if env_metadata:
        discovery_meta["environment_factory"] = env_metadata
    agent_metadata = _build_factory_metadata(targets.agent)
    if agent_metadata:
        discovery_meta["agent_factory"] = agent_metadata
    if llm_provider or llm_model or llm_source:
        discovery_meta["llm"] = {
            "provider": llm_provider,
            "model": llm_model,
            "source": llm_source,
        }
    runtime_meta_info = {
        "environment": targets.environment.dotted_path(),
        "agent": targets.agent.dotted_path(),
        "control_loop": capabilities.get("control_loop", "self"),
        "supports_stepwise": bool(capabilities.get("supports_stepwise", False)),
        "preferred_mode": capabilities.get("preferred_mode", "auto"),
        "behavior": capabilities.get("control_loop", "self"),
        "forced_mode": capabilities.get("forced_mode", capabilities.get("preferred_mode", "auto")),
    }
    metadata_block = copy.deepcopy(metadata_block_current)
    if discovery_meta:
        metadata_block["discovery"] = discovery_meta
    metadata_block.setdefault("runtime", {}).update(runtime_meta_info)
    config_payload["metadata"] = metadata_block

    info = {
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "llm_source": llm_source,
        "llm_inferred": bool(llm_provider or llm_model),
    }
    return config_payload, info


def _prepare_full_config_payload(
    project_root: Path,
    targets: SelectedTargets,
    capabilities: dict[str, object],
    runtime_metadata: dict[str, dict[str, object]] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    template_payload, template_info = _load_full_config_template()
    info = dict(template_info)
    if template_payload is None:
        return None, info
    llm_capabilities = capabilities.get("llm") if isinstance(capabilities, dict) else {}
    if not isinstance(llm_capabilities, dict):
        llm_capabilities = {}
    payload, compose_info = _compose_full_config_payload(
        template_payload,
        targets,
        project_root,
        llm_capabilities,
        capabilities,
        runtime_metadata,
    )
    info.update(compose_info or {})
    if payload is None:
        info.setdefault("reason", compose_info.get("reason") if compose_info else "full-config-unavailable")
    else:
        info.setdefault("mode", "full")
    return payload, info


def _write_stub_config(
    destination: Path,
    targets: SelectedTargets,
    capabilities: dict[str, object],
    runtime_metadata: dict[str, dict[str, object]] | None,
) -> None:
    control_loop = capabilities.get("control_loop", "self")
    supports_stepwise = bool(capabilities.get("supports_stepwise", False))
    preferred_mode = capabilities.get("preferred_mode", "auto")
    plan_description = capabilities.get("plan_description") or ""
    lines: list[str] = [
        "# Generated by `atlas env init`.",
        "# Merge these values into your primary Atlas config or tweak them in-place.",
        "# Secrets (OPENAI_API_KEY, GEMINI_API_KEY, etc.) should live in your shell or .env.",
        "# Set STORAGE__DATABASE_URL if you want to persist learning updates (see README for details).",
        "runtime:",
        "  behavior: self",
        f"  environment: {targets.environment.dotted_path()}",
        f"  agent: {targets.agent.dotted_path()}",
        f"  control_loop: {control_loop}",
        f"  supports_stepwise: {str(supports_stepwise).lower()}",
        f"  preferred_mode: {preferred_mode}",
    ]
    agent_runtime_meta = runtime_metadata.get("agent") if runtime_metadata else {}
    prompt_literals = agent_runtime_meta.get("prompts") if isinstance(agent_runtime_meta, dict) else []
    if isinstance(prompt_literals, list) and prompt_literals:
        prompt = prompt_literals[0]
        if prompt:
            lines.append("  system_prompt: |")
            for line in prompt.splitlines() or [""]:
                lines.append(f"    {line}")
    tool_entries = agent_runtime_meta.get("tools") if isinstance(agent_runtime_meta, dict) else []
    if isinstance(tool_entries, list) and tool_entries:
        lines.append("  tools:")
        for entry in tool_entries:
            if isinstance(entry, dict) and entry.get("name"):
                lines.append(f"    - name: {entry['name']}")
            elif isinstance(entry, str):
                lines.append(f"    - name: {entry}")
    if plan_description:
        indented_plan = plan_description.replace(chr(10), chr(10) + "    ")
        lines.append("  plan_description: |")
        lines.append(f"    {indented_plan}")
    lines.extend(
        [
            "orchestration:",
            f"  forced_mode: {preferred_mode}",
            "storage:",
            '  database_url: ""  # Optional: set to postgresql://user:pass@host:port/dbname or rely on STORAGE__DATABASE_URL',
        ]
    )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_generated_config(
    destination: Path,
    project_root: Path,
    targets: SelectedTargets,
    capabilities: dict[str, object],
    runtime_metadata: dict[str, dict[str, object]] | None,
    *,
    force: bool,
    scaffold_full: bool,
) -> dict[str, Any]:
    _ensure_write(destination, force=force)
    if scaffold_full:
        payload, info = _prepare_full_config_payload(project_root, targets, capabilities, runtime_metadata)
        if payload is not None and yaml is not None:
            rendered = yaml.safe_dump(payload, sort_keys=False)
            destination.write_text(rendered, encoding="utf-8")
            info.setdefault("mode", "full")
            return info
        fallback_info = info if isinstance(info, dict) else {}
    else:
        fallback_info = {}
    _write_stub_config(destination, targets, capabilities, runtime_metadata)
    fallback_info.setdefault("mode", "stub")
    return fallback_info


def _compose_metadata(
    project_root: Path,
    targets: SelectedTargets,
    *,
    discovery_payload: dict[str, object],
    preflight_notes: list[str] | None = None,
    auto_skip: bool = False,
    synthesis_notes: list[str] | None = None,
) -> dict[str, object]:
    generated_at = datetime.now(timezone.utc).isoformat()
    final_answer = discovery_payload.get("final_answer")
    has_final_answer = isinstance(final_answer, str) and final_answer.strip() != ""
    history = discovery_payload.get("history") or []
    agent_caps = targets.agent.candidate.capabilities if targets.agent.candidate else {}
    plan_preview = discovery_payload.get("plan")
    if isinstance(plan_preview, (dict, list)):
        pretty_plan = json.dumps(plan_preview, indent=2)
    elif plan_preview is None:
        pretty_plan = ""
    else:
        pretty_plan = str(plan_preview)
    fallback_control_loop = "self" if agent_caps.get("summarize") else "tool"
    control_loop = "self" if has_final_answer else fallback_control_loop
    fallback_stepwise = bool(agent_caps.get("act"))
    capabilities = {
        "control_loop": control_loop,
        "supports_stepwise": bool(history) if history else fallback_stepwise,
        "plan_description": pretty_plan,
        "telemetry_agent_emitted": False,
        "preferred_mode": "auto" if control_loop == "self" else "paired",
    }
    discovery_capabilities = discovery_payload.get("capabilities")
    if isinstance(discovery_capabilities, dict):
        llm_caps = discovery_capabilities.get("llm")
        if isinstance(llm_caps, dict):
            capabilities["llm"] = {
                "provider": llm_caps.get("provider"),
                "model": llm_caps.get("model"),
                "source": llm_caps.get("source"),
                "inferred": bool(llm_caps.get("provider") or llm_caps.get("model")),
            }
    telemetry_payload = discovery_payload.get("telemetry") or {}
    telemetry_flag = bool(telemetry_payload.get("agent_emitted"))
    capabilities["telemetry_agent_emitted"] = telemetry_flag
    metadata = {
        "version": 1,
        "generated_at": generated_at,
        "project_root": str(project_root),
        "environment": _serialize_target(targets.environment, project_root, "environment"),
        "agent": _serialize_target(targets.agent, project_root, "agent"),
        "capabilities": capabilities,
        "schema": discovery_payload.get("schema") or {},
        "reward": discovery_payload.get("reward") or {},
        "telemetry": telemetry_payload,
        "sample_history": discovery_payload.get("history") or [],
        "plan_preview": plan_preview,
        "final_answer_sample": final_answer,
        "preflight": {
            "notes": preflight_notes or [],
            "auto_skip": auto_skip,
        },
    }
    if synthesis_notes:
            metadata["synthesis"] = {"notes": synthesis_notes}
    return metadata


def _cmd_env_init(args: argparse.Namespace) -> int:
    project_root = Path(args.path or ".").resolve()
    print("Scanning repository...")
    print()
    candidates = discover_candidates(project_root)
    env_candidates, agent_candidates = split_candidates(candidates)

    targets = SelectedTargets(environment=TargetSpec(), agent=TargetSpec())
    atlas_dir = project_root / ".atlas"
    atlas_dir.mkdir(parents=True, exist_ok=True)
    marker_path = atlas_dir / VALIDATION_MARKER_FILENAME
    try:
        marker_path.unlink()
    except FileNotFoundError:
        pass

    try:
        env_kw_pairs = parse_key_value_flags(args.env_kwargs or [])
        agent_kw_pairs = parse_key_value_flags(args.agent_kwargs or [])
    except CLIError as exc:
        print(exc, file=sys.stderr)
        return 1

    env_config_payload: dict[str, object] | None = None
    agent_config_payload: dict[str, object] | None = None
    if args.env_config:
        try:
            env_config_payload = load_config_file(args.env_config)
        except CLIError as exc:
            print(exc, file=sys.stderr)
            return 1
        if not isinstance(env_config_payload, dict):
            print(f"Environment config {args.env_config!r} must be a mapping.", file=sys.stderr)
            return 1
    if args.agent_config:
        try:
            agent_config_payload = load_config_file(args.agent_config)
        except CLIError as exc:
            print(exc, file=sys.stderr)
            return 1
        if not isinstance(agent_config_payload, dict):
            print(f"Agent config {args.agent_config!r} must be a mapping.", file=sys.stderr)
            return 1

    environment_kwargs: dict[str, object] = {}
    agent_kwargs: dict[str, object] = {}
    if env_config_payload:
        environment_kwargs.update(env_config_payload)
    environment_kwargs.update(env_kw_pairs)
    if agent_config_payload:
        agent_kwargs.update(agent_config_payload)
    agent_kwargs.update(agent_kw_pairs)

    targets.environment.kwargs = environment_kwargs
    targets.environment.config = env_config_payload
    targets.agent.kwargs = agent_kwargs
    targets.agent.config = agent_config_payload

    if args.env_fn:
        try:
            targets.environment.factory = parse_callable_reference(args.env_fn)
        except CLIError as exc:
            print(exc, file=sys.stderr)
            return 1
    if args.agent_fn:
        try:
            targets.agent.factory = parse_callable_reference(args.agent_fn)
        except CLIError as exc:
            print(exc, file=sys.stderr)
            return 1

    env_wrapper_required = False
    agent_wrapper_required = False

    auto_messages: list[str] = []
    env_candidate_requires_adapter = False
    agent_candidate_requires_adapter = False
    constructor_gap_notes: list[str] = []

    if targets.environment.factory is None:
        preferred_env_module = targets.environment.factory[0] if targets.environment.factory else None
        auto_candidate, auto_applied = _auto_select_candidate(env_candidates, preferred_env_module)
        if auto_applied and auto_candidate is not None:
            targets.environment.candidate = auto_candidate
            auto_messages.append(f"Auto-selected environment candidate {auto_candidate.dotted_path()} (score={auto_candidate.score}).")
        elif env_candidates:
            try:
                targets.environment.candidate = _prompt_selection(_filter_candidates(env_candidates), "environment")
            except ValueError as exc:
                print(exc, file=sys.stderr)
                return 1
        else:
            # Suppress hint in non-verbose mode - synthesis will auto-generate
            if getattr(args, 'verbose', False):
                _print_factory_hint("environment")
            env_wrapper_required = True
    else:
        env_candidates = []

    if targets.agent.factory is None:
        preferred_agent_module = targets.agent.factory[0] if targets.agent.factory else None
        auto_candidate, auto_applied = _auto_select_candidate(agent_candidates, preferred_agent_module)
        if auto_applied and auto_candidate is not None:
            targets.agent.candidate = auto_candidate
            auto_messages.append(f"Auto-selected agent candidate {auto_candidate.dotted_path()} (score={auto_candidate.score}).")
        elif agent_candidates:
            try:
                targets.agent.candidate = _prompt_selection(_filter_candidates(agent_candidates), "agent")
            except ValueError as exc:
                print(exc, file=sys.stderr)
                return 1
        else:
            # Suppress hint in non-verbose mode - synthesis will auto-generate
            if getattr(args, 'verbose', False):
                _print_factory_hint("agent")
            agent_wrapper_required = True
    else:
        agent_candidates = []

    if targets.environment.candidate is not None:
        targets.environment.metadata = collect_runtime_metadata(project_root, targets.environment.candidate)
    if targets.agent.candidate is not None:
        targets.agent.metadata = collect_runtime_metadata(project_root, targets.agent.candidate)

    def _note_required_params(metadata: dict[str, object] | None, role: str, *, only_if_factory: bool) -> None:
        if not metadata:
            return
        if only_if_factory and not metadata.get("is_factory"):
            return
        params = metadata.get("parameters")
        if not isinstance(params, list):
            return
        required = [param["name"] for param in params if isinstance(param, dict) and param.get("required")]
        if required:
            constructor_gap_notes.append(f"{role.title()} requires: {', '.join(required)}")

    def _metadata_for_role(target: TargetSpec) -> dict[str, object] | None:
        if target.metadata is None:
            return None
        enriched = dict(target.metadata)
        enriched["is_factory"] = bool(target.candidate and target.candidate.is_factory)
        if target.candidate and target.candidate.factory_kind:
            enriched["factory_kind"] = target.candidate.factory_kind
        return enriched

    _note_required_params(_metadata_for_role(targets.environment), "environment", only_if_factory=True)
    _note_required_params(_metadata_for_role(targets.agent), "agent", only_if_factory=True)

    synthesis_notes: list[str] = []
    synthesis_preflight: list[str] = []
    synthesis_auto_skip = False
    synthesizer: FactorySynthesizer | None = None
    llm_synthesis_used = False
    env_missing_required: list[str] = []
    agent_missing_required: list[str] = []

    def _ensure_synthesizer() -> FactorySynthesizer:
        nonlocal synthesizer
        if synthesizer is None:
            synthesizer = FactorySynthesizer(project_root, atlas_dir)
        return synthesizer

    auto_snippets: dict[str, FactorySnippet] = {}
    manual_env_snippet = False
    manual_agent_snippet = False
    generated_factories_path = atlas_dir / "generated_factories.py"
    if (
        targets.environment.candidate is not None
        and targets.environment.factory is None
        and not targets.environment.candidate.is_factory
    ):
        synth = _ensure_synthesizer()
        needs_factory, env_missing_required = synth.needs_factory_for_candidate(
            targets.environment.candidate,
            targets.environment.kwargs,
        )
        if needs_factory:
            env_candidate_requires_adapter = True
            if env_missing_required:
                constructor_gap_notes.append(
                    f"Environment requires defaults for: {', '.join(env_missing_required)}"
                )
                synthesis_notes.append(
                    f"Environment constructor expects: {', '.join(env_missing_required)}"
                )
    if (
        targets.agent.candidate is not None
        and targets.agent.factory is None
        and not targets.agent.candidate.is_factory
    ):
        synth = _ensure_synthesizer()
        needs_factory, agent_missing_required = synth.needs_factory_for_candidate(
            targets.agent.candidate,
            targets.agent.kwargs,
        )
        if needs_factory:
            agent_candidate_requires_adapter = True
            if agent_missing_required:
                constructor_gap_notes.append(
                    f"Agent requires defaults for: {', '.join(agent_missing_required)}"
                )
                synthesis_notes.append(
                    f"Agent constructor expects: {', '.join(agent_missing_required)}"
                )
            if agent_missing_required and not agent_wrapper_required:
                agent_candidate_requires_adapter = False
    if (
        targets.environment.factory is None
        and targets.environment.candidate is not None
        and not env_candidate_requires_adapter
    ):
        env_snippet = build_environment_factory_snippet(
            targets.environment.candidate,
            targets.environment.kwargs,
        )
        auto_snippets["environment"] = env_snippet
        targets.environment.factory = (GENERATED_MODULE, env_snippet.function_name)
        manual_env_snippet = True
        env_candidate_requires_adapter = False
    if (
        targets.agent.factory is None
        and targets.agent.candidate is not None
        and not agent_candidate_requires_adapter
    ):
        agent_snippet = build_agent_factory_snippet(
            targets.agent.candidate,
            targets.agent.kwargs,
        )
        auto_snippets["agent"] = agent_snippet
        targets.agent.factory = (GENERATED_MODULE, agent_snippet.function_name)
        manual_agent_snippet = True
        agent_candidate_requires_adapter = False
    if auto_snippets:
        synthesizer = _ensure_synthesizer()
        synthesizer.emit_manual_snippets(auto_snippets)
        for snippet in auto_snippets.values():
            synthesis_notes.extend(snippet.notes)
        try:
            display_path = generated_factories_path.relative_to(project_root)
        except ValueError:
            display_path = generated_factories_path
        synthesis_notes.append(
            f"Generated factories saved to {display_path}. Review defaults before production use."
        )

    env_needs_factory = targets.environment.factory is None and (
        targets.environment.candidate is not None or env_wrapper_required
    )
    agent_needs_factory = targets.agent.factory is None and (
        targets.agent.candidate is not None or agent_wrapper_required
    )
    if env_candidate_requires_adapter:
        env_needs_factory = True
    if agent_candidate_requires_adapter:
        agent_needs_factory = True
    needs_synthesis = env_needs_factory or agent_needs_factory

    if needs_synthesis:
        environment_summary = None
        agent_summary = None
        try:
            synthesizer = _ensure_synthesizer()
            if env_wrapper_required:
                environment_summary = synthesizer.prepare_repository_summary(
                    "environment",
                    provided_kwargs=targets.environment.kwargs,
                )
            if agent_wrapper_required:
                agent_summary = synthesizer.prepare_repository_summary(
                    "agent",
                    provided_kwargs=targets.agent.kwargs,
                )
            outcome = synthesizer.synthesise(
                environment=(
                    targets.environment.candidate
                    if targets.environment.candidate is not None
                    and not env_wrapper_required
                    and not manual_env_snippet
                    else None
                ),
                agent=(
                    targets.agent.candidate
                    if targets.agent.candidate is not None
                    and not agent_wrapper_required
                    and not manual_agent_snippet
                    else None
                ),
                environment_kwargs=targets.environment.kwargs,
                agent_kwargs=targets.agent.kwargs,
                environment_summary=environment_summary,
                agent_summary=agent_summary,
            )
        except CLIError as exc:
            print(f"Factory synthesis failed: {exc}", file=sys.stderr)
            return 1
        if outcome.environment_factory:
            targets.environment.factory = outcome.environment_factory
        elif env_wrapper_required:
            print("Factory synthesis did not produce an environment wrapper.", file=sys.stderr)
            return 1
        if outcome.agent_factory:
            targets.agent.factory = outcome.agent_factory
        elif agent_wrapper_required:
            print("Factory synthesis did not produce an agent wrapper.", file=sys.stderr)
            return 1
        if outcome.preflight_notes:
            synthesis_preflight.extend(outcome.preflight_notes)
        if outcome.auxiliary_notes:
            synthesis_notes.extend(outcome.auxiliary_notes)
        synthesis_auto_skip = outcome.auto_skip
        if outcome.environment_auto_wrapped:
            targets.environment.auto_wrapped = True
            synthesis_notes.insert(0, "Auto-generated Atlas environment wrapper from repository context.")
        if outcome.agent_auto_wrapped:
            targets.agent.auto_wrapped = True
            synthesis_notes.insert(0, "Auto-generated Atlas agent wrapper from repository context.")
        llm_synthesis_used = True

    skip_reasons = _infer_skip_reasons(targets)
    if targets.environment.auto_wrapped:
        skip_reasons.append(
            "Environment wrapper synthesized automatically; review generated factory before validation."
        )
    if targets.agent.auto_wrapped:
        skip_reasons.append(
            "Agent wrapper synthesized automatically; review generated factory before validation."
        )
    if constructor_gap_notes:
        synthesis_preflight.extend(
            note for note in constructor_gap_notes if note not in synthesis_preflight
        )
    if synthesis_preflight:
        skip_reasons.extend(synthesis_preflight)

    auto_skip = bool(synthesis_auto_skip or skip_reasons)

    discovery_path = atlas_dir / DISCOVERY_FILENAME
    config_path = atlas_dir / GENERATED_CONFIG_FILENAME
    project_env = _load_project_env(project_root)
    try:
        env_overrides = parse_env_flags(args.env_vars or [])
    except CLIError as exc:
        print(exc, file=sys.stderr)
        return 1
    loaded_env_keys: list[str] = []
    for key, value in project_env.items():
        if key not in env_overrides and key not in {ENV_VALIDATE_FLAG}:
            env_overrides[key] = value
            loaded_env_keys.append(key)

    # Apply .env variables to os.environ for LLM synthesis
    # This ensures FactorySynthesizer's LLMClient can access API keys
    for key, value in env_overrides.items():
        if key not in os.environ:
            os.environ[key] = value

    pythonpath_entries = _prepare_pythonpath_overrides(project_root, targets)
    existing_pythonpath = env_overrides.get("PYTHONPATH") or project_env.get("PYTHONPATH") or os.environ.get("PYTHONPATH")
    pythonpath_added: list[str] = []
    if pythonpath_entries:
        base_existing_parts = (
            [part for part in (existing_pythonpath or "").split(os.pathsep) if part]
            if existing_pythonpath
            else []
        )
        combined: list[str] = []
        seen_paths: set[str] = set()
        for candidate in pythonpath_entries + base_existing_parts:
            path_value = candidate.strip()
            if not path_value:
                continue
            if path_value not in seen_paths:
                seen_paths.add(path_value)
                combined.append(path_value)
                if path_value not in base_existing_parts:
                    pythonpath_added.append(path_value)
        if combined:
            env_overrides["PYTHONPATH"] = os.pathsep.join(combined)
    bootstrap_notes: list[str] = []
    if loaded_env_keys:
        preview = ", ".join(sorted(loaded_env_keys)[:5])
        if len(loaded_env_keys) > 5:
            preview += ", …"
        bootstrap_notes.append(f"Loaded .env variables into discovery worker: {preview}")
    if pythonpath_added:
        preview = ", ".join(pythonpath_added[:5])
        if len(pythonpath_added) > 5:
            preview += ", …"
        bootstrap_notes.append(f"Augmented PYTHONPATH with: {preview}")
    env_overrides.setdefault(ENV_VALIDATE_FLAG, "0")
    spec: dict[str, object] = {
        "project_root": str(project_root),
        "task": args.task,
        "run_discovery": True,
        "env": env_overrides,
    }
    if pythonpath_entries:
        spec["pythonpath"] = pythonpath_entries

    env_overrides["ATLAS_SKIP_VALIDATION"] = "1"

    if targets.environment.candidate is not None:
        env_payload: dict[str, object] = {
            "module": targets.environment.candidate.module,
            "qualname": targets.environment.candidate.qualname,
        }
        if targets.environment.kwargs:
            env_payload["init_kwargs"] = targets.environment.kwargs
        if targets.environment.config is not None and targets.environment.config != targets.environment.kwargs:
            env_payload.setdefault("config", targets.environment.config)
        spec["environment"] = env_payload
    if targets.environment.factory is not None:
        factory_module, factory_qualname = targets.environment.factory
        spec["environment_factory"] = {
            "module": factory_module,
            "qualname": factory_qualname,
            "kwargs": targets.environment.kwargs,
        }

    if targets.agent.candidate is not None:
        agent_payload: dict[str, object] = {
            "module": targets.agent.candidate.module,
            "qualname": targets.agent.candidate.qualname,
        }
        if targets.agent.kwargs:
            agent_payload["init_kwargs"] = targets.agent.kwargs
        if targets.agent.config is not None and targets.agent.config != targets.agent.kwargs:
            agent_payload.setdefault("config", targets.agent.config)
        spec["agent"] = agent_payload
    if targets.agent.factory is not None:
        factory_module, factory_qualname = targets.agent.factory
        spec["agent_factory"] = {
            "module": factory_module,
            "qualname": factory_qualname,
            "kwargs": targets.agent.kwargs,
        }

    # Clean output by default, verbose details with --verbose flag
    verbose = getattr(args, 'verbose', False)

    if verbose:
        print("Discovery summary:")
        print(f"  {_summarise_target(targets.environment, 'Environment')}")
        print(f"  {_summarise_target(targets.agent, 'Agent')}")
        for note in auto_messages:
            print(f"  {note}")
        for note in bootstrap_notes:
            print(f"  {note}")
        if synthesis_notes:
            print("  Synthesis notes:")
            for note in synthesis_notes:
                print(f"    - {note}")
        if skip_reasons:
            print("  Preflight notes:")
            for reason in skip_reasons:
                print(f"    - {reason}")
            if auto_skip:
                print("  Auto-skip enabled: runtime execution deferred until prerequisites are satisfied.")
        print()

    if auto_skip:
        spec["run_discovery"] = False
        spec["skip_import"] = True
    else:
        env_overrides[ENV_VALIDATE_FLAG] = "1"
    atlas_dir.mkdir(parents=True, exist_ok=True)
    if discovery_path.exists() and not args.force:
        print(f"{discovery_path} already exists; use --force to refresh.", file=sys.stderr)
        return 1
    if config_path.exists() and not args.force:
        print(f"{config_path} already exists; use --force to refresh.", file=sys.stderr)
        return 1
    discovery_payload: dict[str, object] | None = None
    attempts = 2 if llm_synthesis_used and synthesizer is not None else 1
    for attempt in range(attempts):
        try:
            discovery_payload = invoke_discovery_worker(spec, timeout=args.timeout or 180)
            break
        except DiscoveryWorkerError as exc:
            if synthesizer is not None and llm_synthesis_used and attempt < attempts - 1:
                synthesizer.retry_with_error(exc.traceback or str(exc))
                continue
            print(f"Discovery worker failed: {exc}", file=sys.stderr)
            for hint in _summarise_failure_hints(exc):
                print(f"  Hint: {hint}", file=sys.stderr)
            return 1
        except CLIError as exc:
            print(f"Discovery worker failed: {exc}", file=sys.stderr)
            return 1
    if discovery_payload is None:
        return 1
    metadata = _compose_metadata(
        project_root,
        targets,
        discovery_payload=discovery_payload,
        preflight_notes=skip_reasons,
        auto_skip=auto_skip,
        synthesis_notes=synthesis_notes,
    )
    persist_discovery_run(
        task=args.task,
        project_root=project_root,
        payload=discovery_payload,
        metadata=metadata,
        source="discovery",
    )
    capabilities = metadata.get("capabilities") if isinstance(metadata.get("capabilities"), dict) else {}
    write_discovery_payload(discovery_path, metadata=metadata)
    runtime_metadata = {
        "environment": targets.environment.metadata or {},
        "agent": targets.agent.metadata or {},
    }
    try:
        want_full_config = bool(getattr(args, "scaffold_config_full", False)) or yaml is not None
        scaffold_info = _write_generated_config(
            config_path,
            project_root,
            targets,
            capabilities,
            runtime_metadata=runtime_metadata,
            force=args.force,
            scaffold_full=want_full_config,
        )
    except FileExistsError as exc:
        print(exc, file=sys.stderr)
        return 1
    if verbose:
        print(f"Discovery metadata written to {discovery_path}")
        if scaffold_info.get("mode") == "full":
            print(f"Generated runnable config written to {config_path}")
            template_path = scaffold_info.get("template_path")
            if template_path:
                print(f"  Template source: {template_path}")
            if scaffold_info.get("llm_inferred"):
                provider = scaffold_info.get("llm_provider") or scaffold_info.get("llm", {}).get("provider")
                model_name = scaffold_info.get("llm_model") or scaffold_info.get("llm", {}).get("model")
                print(
                    "  Inferred LLM metadata: provider={provider} model={model}".format(
                        provider=provider or "unknown",
                        model=model_name or "unknown",
                    )
                )
            else:
                if want_full_config:
                    print("  LLM provider/model not inferred; template defaults retained.")
        else:
            print(f"Generated config stub written to {config_path}")
            if want_full_config:
                reason = scaffold_info.get("reason")
                if reason:
                    print(f"  Unable to scaffold full config ({reason}); wrote discovery stub instead.")
        telemetry_status = "enabled" if capabilities.get("telemetry_agent_emitted") else "missing"
        print(
            "Detected handshake: control_loop={control} supports_stepwise={stepwise} telemetry={telemetry}".format(
                control=capabilities.get("control_loop", "unknown"),
                stepwise=capabilities.get("supports_stepwise", False),
                telemetry=telemetry_status,
            )
        )
        print()

    validation_success, validation_errors = _validate_discovered_artifacts(
        project_root,
        atlas_dir,
        targets.environment,
        targets.agent,
    )

    # Clean success output
    if not verbose:
        # Show agent/environment that was discovered
        agent_display = targets.agent.candidate.dotted_path() if targets.agent.candidate else (
            f"{targets.agent.factory[0]}:{targets.agent.factory[1]}" if targets.agent.factory else "unknown"
        )

        try:
            config_display_short = config_path.relative_to(project_root)
        except ValueError:
            config_display_short = config_path

        print(f"\u2713 Agent: {agent_display}")
        print(f"\u2713 Config: {config_display_short}")
        print()

        # Show prerequisites if auto_skip is enabled
        if auto_skip and skip_reasons:
            actionable = _filter_actionable_prerequisites(skip_reasons)
            if actionable:
                print("Prerequisites:")
                for reason in actionable[:3]:  # Limit to first 3 for readability
                    print(f"  - {reason}")
                if len(actionable) > 3:
                    print(f"  ... and {len(actionable) - 3} more (use --verbose for full list)")
                print()

    if validation_success:
        print("Validation succeeded for generated factories.")
        if auto_skip:
            print("  Note: Sample run was skipped due to missing prerequisites.")
            print("  Run atlas again once dependencies are ready.")
    else:
        print("Validation failed; generated factories may not execute correctly:", file=sys.stderr)
        if verbose:
            for message in validation_errors:
                print(f"  - {message}", file=sys.stderr)
        if auto_skip:
            print("  Note: This may be due to missing prerequisites noted above.", file=sys.stderr)
    print()

    marker_path.write_text(
        json.dumps({"validated_at": datetime.now(timezone.utc).isoformat()}, ensure_ascii=False),
        encoding="utf-8",
    )
    try:
        config_display = config_path.relative_to(project_root)
    except ValueError:
        config_display = config_path
    next_command = ["atlas", "run"]
    cwd = Path.cwd().resolve()
    if project_root != cwd:
        next_command.extend(["--path", str(project_root)])
    next_command.extend(["--config", str(config_display)])
    if args.task:
        next_command.extend(["--task", args.task])
    quoted_command = " ".join(shlex.quote(part) for part in next_command)

    # Prominent "Next:" section
    print("Next:")
    if project_root != cwd:
        print(f"  cd {shlex.quote(str(project_root))}")
    print(f"  {quoted_command}")
    return 0


def register_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    env_parser = subparsers.add_parser("env", help="Environment onboarding commands.")
    env_parser.set_defaults(handler=lambda inner_args: env_parser.print_help() or 0)
    env_subparsers = env_parser.add_subparsers(dest="env_command", metavar="<command>")

    init_parser = env_subparsers.add_parser("init", help="Discover Atlas-compatible environments and agents.")
    init_parser.add_argument("--path", default=".", help="Project root to scan for candidates.")
    init_parser.add_argument("--task", default="Sample investigation prompt", help="Sample task to execute during discovery.")
    init_parser.add_argument(
        "--env",
        dest="env_vars",
        metavar="KEY=VALUE",
        action="append",
        default=[],
        help="Environment variable(s) to expose to the worker.",
    )
    init_parser.add_argument(
        "--no-run",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    init_parser.add_argument(
        "--skip-sample-run",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    init_parser.add_argument(
        "--env-fn",
        help="Optional factory callable (module:qualname) to instantiate the environment.",
    )
    init_parser.add_argument(
        "--agent-fn",
        help="Optional factory callable (module:qualname) to instantiate the agent.",
    )
    init_parser.add_argument(
        "--env-arg",
        dest="env_kwargs",
        metavar="KEY=VALUE",
        action="append",
        default=[],
        help="Keyword argument for the environment factory (repeatable).",
    )
    init_parser.add_argument(
        "--agent-arg",
        dest="agent_kwargs",
        metavar="KEY=VALUE",
        action="append",
        default=[],
        help="Keyword argument for the agent factory (repeatable).",
    )
    init_parser.add_argument(
        "--scaffold-config-full",
        action="store_true",
        help=(
            "Generate a runnable Atlas configuration using discovery metadata and template defaults. "
            "Falls back to the legacy stub if the template cannot be loaded."
        ),
    )
    init_parser.add_argument(
        "--env-config",
        help="Path to JSON/YAML file with additional environment factory kwargs.",
    )
    init_parser.add_argument(
        "--agent-config",
        help="Path to JSON/YAML file with additional agent factory kwargs.",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing discovery artefacts under .atlas/.",
    )
    init_parser.add_argument(
        "--timeout",
        type=int,
        default=240,
        help="Timeout (seconds) for the discovery worker (default: %(default)s).",
    )
    init_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed discovery information (synthesis notes, capabilities, validation details).",
    )
    init_parser.set_defaults(handler=_cmd_env_init)

    scaffold_parser = env_subparsers.add_parser(
        "scaffold",
        help="Copy starter factory helpers (e.g., LangGraph adapter) into your project.",
    )
    scaffold_parser.add_argument(
        "--template",
        default="langgraph",
        choices=sorted(SCAFFOLD_TEMPLATES),
        help="Template to scaffold (default: %(default)s).",
    )
    scaffold_parser.add_argument(
        "--output",
        help="Destination file for the scaffold (defaults to template filename).",
    )
    scaffold_parser.add_argument(
        "--path",
        default=".",
        help="Project root where the scaffold should be written (default: current directory).",
    )
    scaffold_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination file if it already exists.",
    )
    scaffold_parser.set_defaults(handler=_cmd_env_scaffold)
