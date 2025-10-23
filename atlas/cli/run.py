"""Runtime helper command consuming discovery metadata."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from atlas.cli.utils import CLIError, execute_runtime, parse_env_flags, write_run_record
from atlas.core import arun as atlas_arun
from atlas.runtime.orchestration.execution_context import ExecutionContext
from atlas.utils.env import load_dotenv_if_available
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
    if not expected_hash or not rel_path:
        return
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


def _ensure_jsonable(value: Any, depth: int = 0) -> Any:
    if depth > 10:
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _ensure_jsonable(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_ensure_jsonable(item, depth + 1) for item in value]
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
        except Exception:
            return str(value)
        return _ensure_jsonable(dumped, depth + 1)
    if hasattr(value, "to_dict"):
        try:
            dumped = value.to_dict()
        except Exception:
            return str(value)
        return _ensure_jsonable(dumped, depth + 1)
    if hasattr(value, "__dict__"):
        return _ensure_jsonable(vars(value), depth + 1)
    return str(value)


def _run_with_config(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        return 1
    project_root = Path(args.path or ".").resolve()
    atlas_dir = project_root / ".atlas"
    atlas_dir.mkdir(parents=True, exist_ok=True)
    load_dotenv_if_available(project_root / ".env")
    sys_path_candidates = [project_root]
    src_dir = project_root / "src"
    if src_dir.exists():
        sys_path_candidates.append(src_dir)
    for candidate in reversed(sys_path_candidates):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
    if getattr(args, "mode", None):
        print("[atlas run] --mode override is not yet supported; using configuration defaults.", file=sys.stderr)
    if getattr(args, "max_steps", None):
        print("[atlas run] --max-steps option is not currently supported.", file=sys.stderr)

    async def _invoke() -> tuple[Any, dict[str, Any]]:
        execution_context = ExecutionContext.get()
        execution_context.reset()
        result = await atlas_arun(
            args.task,
            str(config_path),
            stream_progress=False,
            session_metadata={"source": "atlas run"},
        )
        metadata_snapshot: dict[str, Any] = dict(execution_context.metadata)
        return result, metadata_snapshot

    try:
        result, metadata_snapshot = asyncio.run(_invoke())
    except Exception as exc:  # pragma: no cover - runtime failures surface to CLI
        print(f"Runtime worker failed: {exc}", file=sys.stderr)
        return 1

    metadata = _ensure_jsonable(metadata_snapshot)
    print(f"[atlas run] metadata keys captured: {list(metadata.keys())}", file=sys.stderr)
    ExecutionContext.get().reset()

    run_payload = {
        "task": args.task,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "result": result.model_dump(),
        "metadata": metadata,
    }
    run_path = write_run_record(atlas_dir, run_payload)

    final_answer = result.final_answer if isinstance(result.final_answer, str) else None
    if final_answer and final_answer.strip():
        print("\n=== Final Answer ===")
        print(final_answer.strip())
    else:
        print("\nNo final answer produced. Inspect telemetry for details.")

    steps = metadata.get("steps") if isinstance(metadata, dict) else {}
    attempt_count = 0
    if isinstance(steps, dict):
        for entry in steps.values():
            if isinstance(entry, dict):
                attempts = entry.get("attempts", [])
                if isinstance(attempts, list):
                    attempt_count += len(attempts)
    print(f"\nTelemetry steps captured: {len(steps) if isinstance(steps, dict) else 0} (attempts={attempt_count})")
    print(f"Run artefact saved to {run_path}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    if getattr(args, "config", None):
        return _run_with_config(args)
    project_root = Path(args.path or ".").resolve()
    atlas_dir = project_root / ".atlas"
    metadata_path = atlas_dir / DISCOVERY_FILENAME
    try:
        metadata = _load_metadata(metadata_path)
    except CLIError as exc:
        print(exc, file=sys.stderr)
        return 1
    project_root_value = metadata.get("project_root")
    if isinstance(project_root_value, Path):
        metadata_root = project_root_value.resolve()
    elif isinstance(project_root_value, str):
        metadata_root = Path(project_root_value).resolve()
    else:
        metadata_root = project_root
    env_payload = metadata.get("environment")
    agent_payload = metadata.get("agent")
    preflight = metadata.get("preflight")
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
    if isinstance(preflight, dict):
        notes = preflight.get("notes")
        if notes:
            print("Preflight notes from discovery:")
            for note in notes:
                print(f"  - {note}")
    capabilities_value = metadata.get("capabilities")
    capabilities: Dict[str, object] = capabilities_value if isinstance(capabilities_value, dict) else {}

    def _build_target(target_payload: dict[str, object]) -> tuple[dict[str, object] | None, dict[str, object] | None]:
        raw_kwargs = target_payload.get("kwargs")
        init_kwargs: dict[str, object] = dict(raw_kwargs) if isinstance(raw_kwargs, dict) else {}
        config_payload = target_payload.get("config")
        base_entry: dict[str, object] | None = None
        factory_entry: dict[str, object] | None = None
        module = target_payload.get("module")
        qualname = target_payload.get("qualname")
        module_str = module if isinstance(module, str) else None
        qualname_str = qualname if isinstance(qualname, str) else None
        if module_str and qualname_str:
            base_entry = {
                "module": module_str,
                "qualname": qualname_str,
            }
            if init_kwargs:
                base_entry["init_kwargs"] = init_kwargs
            if config_payload is not None:
                base_entry["config"] = config_payload
        factory_payload = target_payload.get("factory")
        if isinstance(factory_payload, dict):
            factory_module = factory_payload.get("module")
            factory_qualname = factory_payload.get("qualname")
            factory_kwargs = dict(init_kwargs)
            extra_kwargs = factory_payload.get("kwargs")
            if isinstance(extra_kwargs, dict):
                factory_kwargs.update(extra_kwargs)
            if isinstance(factory_module, str) and isinstance(factory_qualname, str):
                factory_entry = {
                    "module": factory_module,
                    "qualname": factory_qualname,
                    "kwargs": factory_kwargs,
                }
        return base_entry, factory_entry

    env_entry, env_factory_entry = _build_target(env_payload)
    agent_entry, agent_factory_entry = _build_target(agent_payload)

    spec: dict[str, object] = {
        "project_root": str(metadata_root),
        "task": args.task,
        "run_discovery": True,
        "env": env_overrides,
    }
    if env_entry:
        spec["environment"] = env_entry
    if env_factory_entry:
        spec["environment_factory"] = env_factory_entry
    if agent_entry:
        spec["agent"] = agent_entry
    if agent_factory_entry:
        spec["agent_factory"] = agent_factory_entry
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
    telemetry_obj = result.get("telemetry")
    telemetry = telemetry_obj if isinstance(telemetry_obj, dict) else {}
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
        "--config",
        default=None,
        help="Path to an Atlas runtime configuration file. When provided, runs the full orchestrator stack instead of the discovery worker.",
    )
    run_parser.add_argument(
        "--mode",
        default=None,
        help="Requested execution mode override (experimental).",
    )
    run_parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of orchestrator steps (currently informational).",
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
