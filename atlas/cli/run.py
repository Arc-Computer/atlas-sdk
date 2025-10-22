"""Runtime helper command consuming discovery metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

from atlas.cli.utils import CLIError, execute_runtime, parse_env_flags
from atlas.sdk.discovery import calculate_file_hash


DISCOVERY_FILENAME = "discover.json"


def _load_metadata(path: Path) -> dict[str, object]:
    if not path.exists():
        raise CLIError(f"Discovery metadata not found at {path}. Run `atlas env init` first.")
    try:
        raw = path.read_text(encoding="utf-8")
        metadata = json.loads(raw)
    except Exception as exc:
        raise CLIError(f"Failed to load discovery metadata: {exc}") from exc
    if not isinstance(metadata, dict):
        raise CLIError("Discovery metadata is malformed.")
    return metadata


def _validate_module_hash(project_root: Path, payload: dict[str, object], role: str) -> None:
    expected_hash = payload.get("hash")
    rel_path = payload.get("file")
    module = payload.get("module")
    if not isinstance(expected_hash, str) or not isinstance(rel_path, str):
        raise CLIError(f"Discovery metadata missing hash for {role}. Re-run `atlas env init`.")
    file_path = project_root / rel_path
    if not file_path.exists():
        raise CLIError(f"{role.title()} module '{module}' not found at {file_path}. Re-run `atlas env init`.")
    current_hash = calculate_file_hash(file_path)
    if current_hash != expected_hash:
        raise CLIError(
            f"{role.title()} module '{module}' has changed since discovery. "
            "Run `atlas env init` again to refresh metadata."
        )


def _cmd_run(args: argparse.Namespace) -> int:
    project_root = Path(args.path or ".").resolve()
    atlas_dir = project_root / ".atlas"
    metadata_path = atlas_dir / DISCOVERY_FILENAME
    try:
        metadata = _load_metadata(metadata_path)
    except CLIError as exc:
        print(exc, file=sys.stderr)
        return 1
    metadata_root = Path(metadata.get("project_root", project_root)).resolve()  # type: ignore[arg-type]
    env_payload = metadata.get("environment")
    agent_payload = metadata.get("agent")
    if not isinstance(env_payload, dict) or not isinstance(agent_payload, dict):
        print("Discovery metadata missing environment/agent payloads. Re-run `atlas env init`.", file=sys.stderr)
        return 1
    try:
        _validate_module_hash(metadata_root, env_payload, "environment")
        _validate_module_hash(metadata_root, agent_payload, "agent")
    except CLIError as exc:
        print(exc, file=sys.stderr)
        return 1
    try:
        env_overrides = parse_env_flags(args.env_vars or [])
    except CLIError as exc:
        print(exc, file=sys.stderr)
        return 1
    capabilities: Dict[str, object] = metadata.get("capabilities", {}) if isinstance(metadata.get("capabilities"), dict) else {}
    spec = {
        "project_root": str(metadata_root),
        "environment": {"module": env_payload.get("module"), "qualname": env_payload.get("qualname")},
        "agent": {"module": agent_payload.get("module"), "qualname": agent_payload.get("qualname")},
        "task": args.task,
        "run_discovery": True,
        "env": env_overrides,
    }
    try:
        result, run_path = execute_runtime(
            spec,
            capabilities=capabilities,
            atlas_dir=atlas_dir,
            task=args.task,
            timeout=args.timeout or 300,
        )
    except CLIError as exc:
        print(f"Runtime worker failed: {exc}", file=sys.stderr)
        return 1
    final_answer = result.get("final_answer")
    if isinstance(final_answer, str) and final_answer.strip():
        print("\n=== Final Answer ===")
        print(final_answer.strip())
    else:
        print("\nNo final answer produced. Inspect telemetry for details.")
    telemetry = result.get("telemetry") or {}
    event_count = len(telemetry.get("events") or [])
    agent_emitted = telemetry.get("agent_emitted", False)
    print(f"\nTelemetry events captured: {event_count}")
    if not agent_emitted:
        print("Agent did not emit telemetry via emit_event; consider instrumenting emit_event calls.")
    print(f"Run artefact saved to {run_path}")
    return 0


def register_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    run_parser = subparsers.add_parser("run", help="Execute the discovered environment/agent pair.")
    run_parser.add_argument("--path", default=".", help="Project root containing .atlas/discover.json.")
    run_parser.add_argument(
        "--env",
        dest="env_vars",
        metavar="KEY=VALUE",
        action="append",
        default=[],
        help="Environment variable(s) to expose to the runtime worker.",
    )
    run_parser.add_argument(
        "--task",
        required=True,
        help="Task prompt to send to the discovered agent.",
    )
    run_parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout (seconds) for the runtime worker (default: %(default)s).",
    )
    run_parser.set_defaults(handler=_cmd_run)
