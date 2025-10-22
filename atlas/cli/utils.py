"""Shared helpers for Atlas CLI commands."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple


class CLIError(RuntimeError):
    """Raised when a CLI helper encounters a recoverable error."""


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
    if process.returncode != 0:
        stderr = process.stderr.strip()
        stdout = process.stdout.strip()
        message = stderr or stdout or f"discovery worker failed with exit code {process.returncode}"
        raise CLIError(message)
    try:
        payload = json.loads(process.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise CLIError(f"Failed to parse discovery worker output: {exc}") from exc
    if payload.get("status") != "ok":
        error = payload.get("error") or "unknown worker error"
        trace = payload.get("traceback")
        if trace:
            print(trace, file=sys.stderr)
        raise CLIError(str(error))
    return payload["result"]  # type: ignore[return-value]


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
    run_record = {
        "task": task,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "capabilities": capabilities,
        "result": result,
    }
    run_path = write_run_record(atlas_dir, run_record)
    return result, run_path
