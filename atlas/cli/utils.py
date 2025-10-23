"""Shared helpers for Atlas CLI commands."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

try:
    import yaml  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore[assignment]

from atlas.cli.persistence import persist_discovery_run


class CLIError(RuntimeError):
    """Raised when a CLI helper encounters a recoverable error."""


class DiscoveryWorkerError(CLIError):
    """Raised when the discovery worker reports an error."""

    def __init__(self, message: str, *, traceback_text: str | None = None) -> None:
        super().__init__(message)
        self.traceback = traceback_text


def parse_env_flags(values: Iterable[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise CLIError(f"Environment flag must be in KEY=VALUE form (received: {item!r})")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise CLIError("Environment variable name cannot be empty.")
        result[key] = value
    return result


def invoke_discovery_worker(spec: dict[str, object], *, timeout: int) -> dict[str, object]:
    process = subprocess.run(
        [sys.executable, "-m", "atlas.sdk.discovery_worker"],
        input=json.dumps(spec),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    stdout = (process.stdout or "").strip()
    stderr = (process.stderr or "").strip()
    payload: dict[str, Any] | None = None
    if stdout:
        try:
            candidate = json.loads(stdout)
            if isinstance(candidate, dict):
                payload = candidate
        except json.JSONDecodeError:
            payload = None
    if payload and payload.get("status") != "ok":
        error = payload.get("error") or "unknown worker error"
        trace = payload.get("traceback")
        if trace:
            print(trace, file=sys.stderr)
        raise DiscoveryWorkerError(str(error), traceback_text=trace)
    if process.returncode != 0:
        message = stderr or stdout or f"discovery worker failed with exit code {process.returncode}"
        raise CLIError(message)
    if not payload:
        raise CLIError("Discovery worker returned no payload.")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise CLIError("Discovery worker result is malformed.")
    return result


def parse_key_value_flags(values: Iterable[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise CLIError(f"Expected KEY=VALUE format (received: {item!r})")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise CLIError("Key cannot be empty.")
        result[key] = value
    return result


def parse_callable_reference(value: str) -> Tuple[str, str]:
    if ":" not in value:
        raise CLIError(f"Expected module:qualname format (received: {value!r})")
    module, qualname = value.split(":", 1)
    module = module.strip()
    qualname = qualname.strip()
    if not module or not qualname:
        raise CLIError(f"Invalid callable reference: {value!r}")
    return module, qualname


def load_config_file(path: str) -> Dict[str, Any]:
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise CLIError(f"Config file not found: {file_path}")
    if file_path.suffix in {".json"}:
        return json.loads(file_path.read_text(encoding="utf-8"))
    if file_path.suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise CLIError("PyYAML is required to load YAML configuration files.")
        return yaml.safe_load(file_path.read_text(encoding="utf-8"))
    raise CLIError(f"Unsupported config extension for {file_path}. Expected .json, .yaml, or .yml.")


def write_run_record(atlas_dir: Path, payload: dict[str, object]) -> Path:
    runs_dir = atlas_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = runs_dir / f"run_{timestamp}.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def execute_runtime(
    spec: dict[str, object],
    *,
    capabilities: Dict[str, object],
    atlas_dir: Path,
    task: str,
    timeout: int,
) -> Tuple[dict[str, object], Path]:
    result = invoke_discovery_worker(spec, timeout=timeout)
    final_answer = result.get("final_answer")
    if capabilities.get("control_loop") == "self" and not (isinstance(final_answer, str) and final_answer.strip()):
        raise CLIError(
            "Agent did not submit a final answer, but discovery marked control_loop=self. "
            "Re-run `atlas env init` to refresh metadata."
        )
    run_record: dict[str, object] = {
        "task": task,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "capabilities": capabilities,
        "result": result,
    }
    run_path = write_run_record(atlas_dir, run_record)
    project_root_value = spec.get("project_root")
    if isinstance(project_root_value, Path):
        project_root = project_root_value.resolve()
    elif isinstance(project_root_value, str):
        project_root = Path(project_root_value).resolve()
    else:
        project_root = Path(".").resolve()
    metadata: dict[str, object] = {
        "capabilities": capabilities,
        "run_artifact": str(run_path),
    }
    persist_discovery_run(
        task=task,
        project_root=project_root,
        payload=result,
        metadata=metadata,
        source="runtime",
    )
    return result, run_path
