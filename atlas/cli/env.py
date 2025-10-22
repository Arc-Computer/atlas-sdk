"""Environment onboarding CLI commands."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from atlas.cli.utils import CLIError, execute_runtime, invoke_discovery_worker, parse_env_flags
from atlas.sdk.discovery import Candidate, discover_candidates, serialize_candidate, split_candidates, write_discovery_payload


DISCOVERY_FILENAME = "discover.json"
GENERATED_CONFIG_FILENAME = "generated_config.yaml"


@dataclass(slots=True)
class SelectedTargets:
    environment: Candidate
    agent: Candidate


def _prompt_selection(candidates: List[Candidate], role: str) -> Candidate:
    if not candidates:
        raise ValueError(f"No candidates detected for role '{role}'.")
    if len(candidates) == 1:
        return candidates[0]
    print(f"Multiple {role} candidates detected:")
    for index, candidate in enumerate(candidates, start=1):
        marker = "[decorator]" if candidate.via_decorator else "[heuristic]"
        print(f"  {index}. {candidate.dotted_path()} {marker} score={candidate.score}")
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


def _ensure_write(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists. Use --force to overwrite.")


def _write_generated_config(destination: Path, targets: SelectedTargets, capabilities: dict[str, object], *, force: bool) -> None:
    _ensure_write(destination, force=force)
    control_loop = capabilities.get("control_loop", "self")
    supports_stepwise = bool(capabilities.get("supports_stepwise", False))
    plan_description = capabilities.get("plan_description") or ""
    payload = "\n".join(
        [
            "runtime:",
            "  behavior: self",
            f"  environment: {targets.environment.dotted_path()}",
            f"  agent: {targets.agent.dotted_path()}",
            f"  control_loop: {control_loop}",
            f"  supports_stepwise: {str(supports_stepwise).lower()}",
        ]
    )
    if plan_description:
        payload += f"\n  plan_description: |\n    {plan_description.replace(chr(10), chr(10) + '    ')}"
    destination.write_text(payload + "\n", encoding="utf-8")


def _compose_metadata(
    project_root: Path,
    targets: SelectedTargets,
    *,
    discovery_payload: dict[str, object],
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
    capabilities = {
        "control_loop": "self" if has_final_answer else "tool",
        "supports_stepwise": False if has_final_answer else bool(history),
        "plan_description": pretty_plan,
        "telemetry_agent_emitted": bool(
            (discovery_payload.get("telemetry") or {}).get("agent_emitted")
        ),
    }
    metadata = {
        "version": 1,
        "generated_at": generated_at,
        "project_root": str(project_root),
        "environment": serialize_candidate(targets.environment, project_root),
        "agent": serialize_candidate(targets.agent, project_root),
        "capabilities": capabilities,
        "schema": discovery_payload.get("schema") or {},
        "reward": discovery_payload.get("reward") or {},
        "telemetry": discovery_payload.get("telemetry") or {},
        "sample_history": discovery_payload.get("history") or [],
        "plan_preview": plan_preview,
        "final_answer_sample": final_answer,
    }
    return metadata


def _cmd_env_init(args: argparse.Namespace) -> int:
    project_root = Path(args.path or ".").resolve()
    candidates = discover_candidates(project_root)
    env_candidates, agent_candidates = split_candidates(candidates)
    if not env_candidates:
        print("No environment candidates discovered. Ensure classes are decorated with @atlas.environment or expose reset/step/close.", file=sys.stderr)
        return 1
    if not agent_candidates:
        print("No agent candidates discovered. Ensure classes are decorated with @atlas.agent or expose plan/act/summarize.", file=sys.stderr)
        return 1
    try:
        env_choice = _prompt_selection(env_candidates, "environment")
        agent_choice = _prompt_selection(agent_candidates, "agent")
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    targets = SelectedTargets(environment=env_choice, agent=agent_choice)
    atlas_dir = project_root / ".atlas"
    discovery_path = atlas_dir / DISCOVERY_FILENAME
    config_path = atlas_dir / GENERATED_CONFIG_FILENAME
    try:
        env_overrides = parse_env_flags(args.env_vars or [])
    except CLIError as exc:
        print(exc, file=sys.stderr)
        return 1
    spec = {
        "project_root": str(project_root),
        "environment": {"module": targets.environment.module, "qualname": targets.environment.qualname},
        "agent": {"module": targets.agent.module, "qualname": targets.agent.qualname},
        "task": args.task,
        "run_discovery": not args.no_run,
        "env": env_overrides,
    }
    atlas_dir.mkdir(parents=True, exist_ok=True)
    if discovery_path.exists() and not args.force:
        print(f"{discovery_path} already exists; use --force to refresh.", file=sys.stderr)
        return 1
    if config_path.exists() and not args.force:
        print(f"{config_path} already exists; use --force to refresh.", file=sys.stderr)
        return 1
    try:
        discovery_payload = invoke_discovery_worker(spec, timeout=args.timeout or 180)
    except CLIError as exc:
        print(f"Discovery worker failed: {exc}", file=sys.stderr)
        return 1
    metadata = _compose_metadata(project_root, targets, discovery_payload=discovery_payload)
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
    if args.no_run:
        print("Run skipped (--no-run supplied).")
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
