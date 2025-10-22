"""Environment onboarding CLI commands."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from atlas.cli.utils import (
    CLIError,
    DiscoveryWorkerError,
    execute_runtime,
    invoke_discovery_worker,
    load_config_file,
    parse_callable_reference,
    parse_env_flags,
    parse_key_value_flags,
)
from atlas.sdk.discovery import Candidate, Role, discover_candidates, serialize_candidate, split_candidates, write_discovery_payload
from atlas.sdk.factory_synthesis import FactorySynthesizer, ENV_VALIDATE_FLAG


DISCOVERY_FILENAME = "discover.json"
GENERATED_CONFIG_FILENAME = "generated_config.yaml"


@dataclass(slots=True)
class TargetSpec:
    candidate: Candidate | None = None
    factory: tuple[str, str] | None = None
    kwargs: Dict[str, object] = field(default_factory=dict)
    config: dict[str, object] | None = None

    def dotted_path(self) -> str:
        if self.candidate is not None:
            return self.candidate.dotted_path()
        if self.factory is not None:
            return f"{self.factory[0]}:{self.factory[1]}"
        return "<unspecified>"


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
    # Ensure required keys exist even if no candidate was discovered.
    payload.setdefault("module", None)
    payload.setdefault("qualname", None)
    payload.setdefault("file", None)
    payload.setdefault("hash", None)
    return payload


@dataclass(slots=True)
class SelectedTargets:
    environment: TargetSpec
    agent: TargetSpec


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


def _write_generated_config(destination: Path, targets: SelectedTargets, capabilities: dict[str, object], *, force: bool) -> None:
    _ensure_write(destination, force=force)
    control_loop = capabilities.get("control_loop", "self")
    supports_stepwise = bool(capabilities.get("supports_stepwise", False))
    preferred_mode = capabilities.get("preferred_mode", "auto")
    plan_description = capabilities.get("plan_description") or ""
    payload = "\n".join(
        [
            "runtime:",
            "  behavior: self",
            f"  environment: {targets.environment.dotted_path()}",
            f"  agent: {targets.agent.dotted_path()}",
            f"  control_loop: {control_loop}",
            f"  supports_stepwise: {str(supports_stepwise).lower()}",
            f"  preferred_mode: {preferred_mode}",
        ]
    )
    if plan_description:
        payload += f"\n  plan_description: |\n    {plan_description.replace(chr(10), chr(10) + '    ')}"
    payload += "\norchestration:\n  forced_mode: {mode}".format(mode=preferred_mode)
    destination.write_text(payload + "\n", encoding="utf-8")


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
    plan_preview = discovery_payload.get("plan")
    if isinstance(plan_preview, (dict, list)):
        pretty_plan = json.dumps(plan_preview, indent=2)
    elif plan_preview is None:
        pretty_plan = ""
    else:
        pretty_plan = str(plan_preview)
    control_loop = "self" if has_final_answer else "tool"
    capabilities = {
        "control_loop": control_loop,
        "supports_stepwise": False if has_final_answer else bool(history),
        "plan_description": pretty_plan,
        "telemetry_agent_emitted": bool(
            (discovery_payload.get("telemetry") or {}).get("agent_emitted")
        ),
        "preferred_mode": "auto" if control_loop == "self" else "paired",
    }
    metadata = {
        "version": 1,
        "generated_at": generated_at,
        "project_root": str(project_root),
        "environment": _serialize_target(targets.environment, project_root, "environment"),
        "agent": _serialize_target(targets.agent, project_root, "agent"),
        "capabilities": capabilities,
        "schema": discovery_payload.get("schema") or {},
        "reward": discovery_payload.get("reward") or {},
        "telemetry": discovery_payload.get("telemetry") or {},
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
    candidates = discover_candidates(project_root)
    env_candidates, agent_candidates = split_candidates(candidates)

    targets = SelectedTargets(environment=TargetSpec(), agent=TargetSpec())

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

    if env_candidates:
        try:
            targets.environment.candidate = _prompt_selection(env_candidates, "environment")
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1
    elif targets.environment.factory is None:
        print(
            "No environment candidates discovered and no factory provided. Supply --env-fn to continue.",
            file=sys.stderr,
        )
        return 1

    if agent_candidates:
        try:
            targets.agent.candidate = _prompt_selection(agent_candidates, "agent")
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1
    elif targets.agent.factory is None:
        print(
            "No agent candidates discovered and no factory provided. Supply --agent-fn to continue.",
            file=sys.stderr,
        )
        return 1

    atlas_dir = project_root / ".atlas"
    run_requested = not args.no_run

    synthesis_notes: list[str] = []
    synthesis_preflight: list[str] = []
    synthesis_auto_skip = False
    synthesizer: FactorySynthesizer | None = None
    synthesis_used = False

    needs_synthesis = (
        (targets.environment.factory is None and targets.environment.candidate is not None)
        or (targets.agent.factory is None and targets.agent.candidate is not None)
    )

    if needs_synthesis:
        try:
            synthesizer = FactorySynthesizer(project_root, atlas_dir)
            outcome = synthesizer.synthesise(
                environment=targets.environment.candidate if targets.environment.factory is None else None,
                agent=targets.agent.candidate if targets.agent.factory is None else None,
                environment_kwargs=targets.environment.kwargs,
                agent_kwargs=targets.agent.kwargs,
            )
        except CLIError as exc:
            print(f"Factory synthesis failed: {exc}", file=sys.stderr)
            return 1
        if outcome.environment_factory:
            targets.environment.factory = outcome.environment_factory
            synthesis_used = True
        if outcome.agent_factory:
            targets.agent.factory = outcome.agent_factory
            synthesis_used = True
        synthesis_preflight = outcome.preflight_notes
        synthesis_notes = outcome.auxiliary_notes
        synthesis_auto_skip = outcome.auto_skip

    skip_reasons = _infer_skip_reasons(targets)
    if synthesis_preflight:
        skip_reasons.extend(synthesis_preflight)

    auto_skip = False
    if synthesis_auto_skip and run_requested and not args.validate:
        auto_skip = True
        run_requested = False
    elif skip_reasons and run_requested and not args.validate:
        auto_skip = True
        run_requested = False

    discovery_path = atlas_dir / DISCOVERY_FILENAME
    config_path = atlas_dir / GENERATED_CONFIG_FILENAME
    try:
        env_overrides = parse_env_flags(args.env_vars or [])
    except CLIError as exc:
        print(exc, file=sys.stderr)
        return 1
    env_overrides.setdefault(ENV_VALIDATE_FLAG, "1" if run_requested else "0")
    spec: dict[str, object] = {
        "project_root": str(project_root),
        "task": args.task,
        "run_discovery": run_requested,
        "env": env_overrides,
    }

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

    print("Discovery summary:")
    print(f"  {_summarise_target(targets.environment, 'Environment')}")
    print(f"  {_summarise_target(targets.agent, 'Agent')}")
    if synthesis_notes:
        print("  Synthesis notes:")
        for note in synthesis_notes:
            print(f"    - {note}")
    if skip_reasons:
        print("  Preflight notes:")
        for reason in skip_reasons:
            print(f"    - {reason}")
        if auto_skip:
            print("  Auto-skip enabled: discovery run deferred (use --validate to execute once prerequisites are met).")
        elif args.validate:
            print("  Proceeding with discovery despite preflight warnings (--validate supplied).")

    if auto_skip:
        spec["skip_import"] = True
    atlas_dir.mkdir(parents=True, exist_ok=True)
    if discovery_path.exists() and not args.force:
        print(f"{discovery_path} already exists; use --force to refresh.", file=sys.stderr)
        return 1
    if config_path.exists() and not args.force:
        print(f"{config_path} already exists; use --force to refresh.", file=sys.stderr)
        return 1
    discovery_payload: dict[str, object] | None = None
    attempts = 2 if synthesis_used and synthesizer is not None else 1
    for attempt in range(attempts):
        try:
            discovery_payload = invoke_discovery_worker(spec, timeout=args.timeout or 180)
            break
        except DiscoveryWorkerError as exc:
            if synthesizer is not None and synthesis_used and attempt < attempts - 1:
                synthesizer.retry_with_error(exc.traceback or str(exc))
                continue
            print(f"Discovery worker failed: {exc}", file=sys.stderr)
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
    capabilities = metadata.get("capabilities") if isinstance(metadata.get("capabilities"), dict) else {}
    write_discovery_payload(discovery_path, metadata=metadata)
    try:
        _write_generated_config(config_path, targets, capabilities, force=args.force)
    except FileExistsError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"Discovery metadata written to {discovery_path}")
    print(f"Generated config stub written to {config_path}")
    telemetry_status = "enabled" if capabilities.get("telemetry_agent_emitted") else "missing"
    print(
        "Detected handshake: control_loop={control} supports_stepwise={stepwise} telemetry={telemetry}".format(
            control=capabilities.get("control_loop", "unknown"),
            stepwise=capabilities.get("supports_stepwise", False),
            telemetry=telemetry_status,
        )
    )
    if not run_requested:
        if args.no_run:
            print("Run skipped (--no-run supplied).")
        elif auto_skip:
            print("Run skipped automatically based on preflight guidance. Use `atlas env init --validate` once dependencies are ready.")
        else:
            print("Run skipped.")
    else:
        print("Discovery loop completed. Review .atlas/discover.json for captured telemetry.")
        if not args.skip_sample_run:
            runtime_spec = dict(spec)
            runtime_spec["run_discovery"] = True
            try:
                result, run_path = execute_runtime(
                    runtime_spec,
                    capabilities=capabilities,
                    atlas_dir=atlas_dir,
                    task=args.task,
                    timeout=args.timeout or 240,
                )
                final_answer = result.get("final_answer")
                print(f"Sample run recorded at {run_path}")
                if isinstance(final_answer, str) and final_answer.strip():
                    print("Sample run final answer:")
                    print(final_answer.strip())
            except CLIError as exc:
                print(f"Sample runtime failed: {exc}", file=sys.stderr)
        else:
            print("Skipping immediate sample run (--skip-sample-run supplied).")
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
        help="Skip executing the discovery loop; only detect candidates.",
    )
    init_parser.add_argument(
        "--skip-sample-run",
        action="store_true",
        help="Skip suggesting the immediate sample run after discovery completes.",
    )
    init_parser.add_argument(
        "--validate",
        action="store_true",
        help="Force execution of the discovery loop even if preflight diagnostics recommend skipping.",
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
    init_parser.set_defaults(handler=_cmd_env_init)
