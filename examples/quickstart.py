"""Atlas quickstart showing continual learning efficiency gains."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from atlas import core
from atlas.runtime.orchestration.execution_context import ExecutionContext

try:
    import litellm
except ImportError:  # pragma: no cover - safety guard
    litellm = None


TASK = "Summarize the latest Atlas SDK updates in three bullet points."
CONFIG_PATH = "configs/examples/openai_agent.yaml"


@dataclass
class PassMetrics:
    label: str
    duration_seconds: float
    tokens: Optional[int]
    success: bool
    metadata: Dict[str, Any]
    token_breakdown: Dict[str, int] = field(default_factory=dict)
    session_id: Optional[int] = None
    adaptive_mode: Optional[str] = None
    certification_run: bool = False
    persona_fingerprint: Optional[str] = None


USAGE_SUMMARY: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}


def reset_usage_tracker() -> None:
    """Reset liteLLM usage aggregation between passes."""
    global USAGE_SUMMARY
    USAGE_SUMMARY = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}


def usage_callback(kwargs: Dict[str, Any], response_obj: Any, start_time: Any, end_time: Any) -> None:
    """Aggregate token usage from liteLLM callbacks."""
    del start_time, end_time  # unused in aggregation

    usage_payload: Any | None = getattr(response_obj, "usage", None)
    if usage_payload is None and isinstance(response_obj, dict):
        usage_payload = response_obj.get("usage")
    if usage_payload is None and isinstance(kwargs, dict):
        standard = kwargs.get("standard_logging_object")
        if isinstance(standard, dict):
            usage_payload = (
                standard.get("response", {}).get("usage")
                or standard.get("usage")
                or kwargs.get("usage")
            )

    if usage_payload is None:
        return

    def _lookup(field: str) -> int:
        value = getattr(usage_payload, field, None)
        if value is None and isinstance(usage_payload, dict):
            value = usage_payload.get(field)
        if isinstance(value, (int, float)):
            return int(value)
        return 0

    prompt_tokens = _lookup("prompt_tokens")
    completion_tokens = _lookup("completion_tokens")
    total_tokens = _lookup("total_tokens")

    if prompt_tokens == 0 and completion_tokens == 0 and total_tokens == 0:
        return

    global USAGE_SUMMARY
    USAGE_SUMMARY["prompt_tokens"] += prompt_tokens
    USAGE_SUMMARY["completion_tokens"] += completion_tokens
    if total_tokens:
        USAGE_SUMMARY["total_tokens"] += total_tokens
    else:
        USAGE_SUMMARY["total_tokens"] += prompt_tokens + completion_tokens
    USAGE_SUMMARY["calls"] += 1


def register_usage_callback() -> None:
    """Ensure the usage aggregation callback is registered exactly once."""
    if litellm is None:
        return
    if usage_callback in getattr(litellm, "success_callback", []):
        return
    litellm.success_callback.append(usage_callback)


def collect_usage_summary() -> Dict[str, int]:
    """Return a copy of the latest usage summary."""
    return dict(USAGE_SUMMARY)


def wait_for_persona_learning(fingerprint: Optional[str], timeout: float = 8.0, interval: float = 0.5) -> str:
    """Poll Postgres until a persona memory row exists for the fingerprint."""
    if not fingerprint:
        return ""
    deadline = time.time() + timeout
    snapshot = ""
    while time.time() < deadline:
        snapshot = fetch_persona_learning(limit=3, fingerprint=fingerprint)
        if snapshot and "(0 rows)" not in snapshot:
            return snapshot
        time.sleep(interval)
    return snapshot


def extract_token_usage(result: Any, metadata: Dict[str, Any]) -> Optional[int]:
    """Best-effort extraction of total tokens from step metadata and reward payloads."""
    totals: list[int] = []
    for step in getattr(result, "step_results", []):
        step_meta = getattr(step, "metadata", {}) or {}
        step_tokens: Optional[int] = None

        usage_info = step_meta.get("usage")
        if isinstance(usage_info, dict):
            total = usage_info.get("total_tokens")
            prompt = usage_info.get("prompt_tokens")
            completion = usage_info.get("completion_tokens")
            if isinstance(total, (int, float)):
                step_tokens = max(step_tokens or 0, int(total))
            else:
                prompt_val = int(prompt) if isinstance(prompt, (int, float)) else 0
                completion_val = int(completion) if isinstance(completion, (int, float)) else 0
                if prompt_val or completion_val:
                    step_tokens = max(step_tokens or 0, prompt_val + completion_val)

        token_info = step_meta.get("token_counts")
        if isinstance(token_info, dict):
            approx = token_info.get("approx_total")
            if isinstance(approx, (int, float)):
                step_tokens = max(step_tokens or 0, int(approx))

        reasoning_entries = step_meta.get("reasoning") or []
        for entry in reasoning_entries:
            if not isinstance(entry, dict):
                continue
            payload = entry.get("payload")
            if not isinstance(payload, dict):
                continue
            token_payload = payload.get("token_counts")
            if isinstance(token_payload, dict):
                approx = token_payload.get("approx_total") or token_payload.get("accumulated")
                if isinstance(approx, (int, float)):
                    step_tokens = max(step_tokens or 0, int(approx))

        if step_tokens is not None:
            totals.append(step_tokens)

    if totals:
        return sum(totals)

    session_reward = metadata.get("session_reward")
    if isinstance(session_reward, dict):
        raw_payload = session_reward.get("raw")
        if isinstance(raw_payload, dict):
            token_usage = raw_payload.get("token_usage") or raw_payload.get("token_counts")
            if isinstance(token_usage, dict):
                total = token_usage.get("total_tokens") or token_usage.get("approx_total")
                if isinstance(total, (int, float)):
                    return int(total)

    return None


def format_tokens(value: Optional[int]) -> str:
    if value is None:
        return "n/a"
    return f"{value:,d}"


def format_token_breakdown(breakdown: Dict[str, int]) -> str:
    if not breakdown:
        return ""
    prompt = breakdown.get("prompt_tokens", 0)
    completion = breakdown.get("completion_tokens", 0)
    calls = breakdown.get("calls", 0)
    return f" (prompt: {prompt:,d}, completion: {completion:,d}, calls: {calls})"


def print_metadata(metadata: Dict[str, Any]) -> None:
    adaptive = metadata.get("adaptive_summary")
    if isinstance(adaptive, dict):
        mode = adaptive.get("adaptive_mode") or metadata.get("execution_mode")
        certification = adaptive.get("certification_run")
        confidence = adaptive.get("confidence") or adaptive.get("probe", {}).get("confidence")
        parts = [f"mode={mode}" if mode else None]
        if confidence is not None:
            parts.append(f"confidence={confidence:.2f}")
        if certification:
            parts.append("certification=True")
        summary_line = ", ".join(part for part in parts if part)
        print("\nAdaptive summary:", summary_line or adaptive)
    reward = metadata.get("reward_summary")
    if isinstance(reward, dict):
        print("Reward summary:", reward)
    else:
        session_reward = metadata.get("session_reward")
        if isinstance(session_reward, dict):
            print("Reward summary:", session_reward)


def execute_pass(header: str) -> PassMetrics:
    print(header)
    reset_usage_tracker()
    start_time = time.time()
    result = core.run(
        task=TASK,
        config_path=CONFIG_PATH,
        stream_progress=True,
    )
    duration = time.time() - start_time

    metadata_snapshot = deepcopy(ExecutionContext.get().metadata)
    usage_summary = metadata_snapshot.get("token_usage") if isinstance(metadata_snapshot, dict) else None

    tokens: Optional[int] = None
    token_breakdown: Dict[str, int] = {}
    if isinstance(usage_summary, dict) and usage_summary.get("calls"):
        token_breakdown = {
            "prompt_tokens": int(usage_summary.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage_summary.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage_summary.get("total_tokens", 0) or 0),
            "calls": int(usage_summary.get("calls", 0) or 0),
        }
        tokens = token_breakdown["total_tokens"]
    else:
        aggregated = collect_usage_summary()
        if aggregated.get("calls"):
            token_breakdown = {
                "prompt_tokens": aggregated.get("prompt_tokens", 0),
                "completion_tokens": aggregated.get("completion_tokens", 0),
                "total_tokens": aggregated.get("total_tokens", 0),
                "calls": aggregated.get("calls", 0),
            }
            tokens = token_breakdown["total_tokens"]

    if tokens is None:
        tokens = extract_token_usage(result, metadata_snapshot)

    success = bool(result.final_answer.strip())
    session_id = fetch_latest_session_id()
    adaptive_meta = metadata_snapshot.get("adaptive_summary")
    persona_fingerprint = metadata_snapshot.get("persona_fingerprint")
    adaptive_mode = None
    certification_run = False
    if isinstance(adaptive_meta, dict):
        adaptive_mode = adaptive_meta.get("adaptive_mode") or metadata_snapshot.get("execution_mode")
        certification_run = bool(adaptive_meta.get("certification_run"))
    elif isinstance(metadata_snapshot.get("adaptive"), dict):
        adaptive_mode = metadata_snapshot["adaptive"].get("active_mode")
        certification_run = bool(metadata_snapshot["adaptive"].get("certification_run"))

    print("\n=== Final Answer ===")
    print(result.final_answer)
    print_metadata(metadata_snapshot)
    breakdown_suffix = format_token_breakdown(token_breakdown)
    print(f"\nExecution time: {duration:.1f}s")
    print(f"Tokens generated: {format_tokens(tokens)}{breakdown_suffix}")
    print(f"Status: {'success' if success else 'failure'}")
    if adaptive_mode:
        status_details = f"Adaptive mode: {adaptive_mode}"
        if certification_run:
            status_details += " (certification)"
        print(status_details)

    return PassMetrics(
        label=header,
        duration_seconds=duration,
        tokens=tokens,
        success=success,
        metadata=metadata_snapshot,
        token_breakdown=token_breakdown,
        session_id=session_id,
        adaptive_mode=adaptive_mode,
        certification_run=certification_run,
        persona_fingerprint=persona_fingerprint,
    )


def print_efficiency(pass_one: PassMetrics, pass_two: PassMetrics) -> None:
    print("\n=== Efficiency Comparison ===")

    if pass_one.duration_seconds > 0:
        time_saved = pass_one.duration_seconds - pass_two.duration_seconds
        pct = (time_saved / pass_one.duration_seconds) * 100
        direction = "faster" if time_saved >= 0 else "slower"
        print(f"Time saved: {abs(time_saved):.1f}s ({abs(pct):.0f}% {direction})")
    else:
        print("Time saved: n/a")

    if pass_one.tokens is not None and pass_two.tokens is not None and pass_one.tokens > 0:
        token_saved = pass_one.tokens - pass_two.tokens
        pct_tokens = (token_saved / pass_one.tokens) * 100
        direction = "reduction" if token_saved >= 0 else "increase"
        print(f"Tokens saved: {abs(token_saved):,d} ({abs(pct_tokens):.0f}% {direction})")
    else:
        print("Tokens saved: n/a")

    fingerprint = pass_two.persona_fingerprint or pass_one.persona_fingerprint
    usage_report = fetch_persona_memory_usage(fingerprint)
    if usage_report:
        if "(0 rows)" in usage_report:
            print("Learning applied: no persona_memory_usage records found for this fingerprint.")
        else:
            if fingerprint:
                prefix = f"Learning applied (fingerprint={fingerprint[:8]}...):"
            else:
                prefix = "Learning applied:"
            print(f"{prefix}\n{usage_report}")
    else:
        print("Learning applied: query returned no output.")

    session_ids = [sid for sid in (pass_one.session_id, pass_two.session_id) if sid]
    student_learning_report = fetch_student_learning(session_ids)
    if student_learning_report:
        print(f"\nSession learning signals:\n{student_learning_report}")

    stored_learning = fetch_persona_learning(
        fingerprint=fingerprint
    )
    if stored_learning:
        if "(0 rows)" in stored_learning:
            print("\nStored learning: no persona_memory rows yet for this fingerprint.")
        else:
            if fingerprint:
                print(f"\nStored learning (fingerprint={fingerprint[:8]}...):\n{stored_learning}")
            else:
                print(f"\nStored learning:\n{stored_learning}")


def run_psql(query: str, *, tuples_only: bool = False) -> str:
    """Execute a SQL query via psql and return stdout."""
    env = os.environ.copy()
    env.setdefault("PGPASSWORD", "atlas")
    args = [
        "psql",
        "-h",
        "localhost",
        "-p",
        "5433",
        "-U",
        "atlas",
        "-d",
        "atlas",
    ]
    if tuples_only:
        args.append("-At")
    args.extend(["-c", query])
    try:
        completed = subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
    except FileNotFoundError:
        return "psql command not available on PATH."
    except subprocess.CalledProcessError as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        return output.strip()
    return completed.stdout.strip()


def fetch_persona_memory_usage(fingerprint: Optional[str]) -> str:
    if fingerprint:
        fingerprint = fingerprint.replace("'", "''")
        return run_psql(
            "SELECT u.memory_id, u.session_id, u.applied_at "
            "FROM persona_memory_usage u "
            "JOIN persona_memory m ON m.memory_id = u.memory_id "
            f"WHERE m.trigger_fingerprint = '{fingerprint}' "
            "ORDER BY u.applied_at DESC LIMIT 5;"
        )
    return run_psql(
        "SELECT memory_id, session_id, applied_at "
        "FROM persona_memory_usage "
        "ORDER BY applied_at DESC LIMIT 5;"
    )


def fetch_persona_learning(limit: int = 3, fingerprint: Optional[str] = None) -> str:
    where_clause = ""
    if fingerprint:
        fingerprint = fingerprint.replace("'", "''")
        where_clause = f"WHERE trigger_fingerprint = '{fingerprint}' "
    return run_psql(
        "SELECT memory_id, instruction::text AS instruction, created_at "
        f"FROM persona_memory {where_clause}ORDER BY created_at DESC LIMIT {limit};"
    )


def fetch_latest_session_id() -> Optional[int]:
    output = run_psql(
        "SELECT id FROM sessions ORDER BY created_at DESC LIMIT 1;",
        tuples_only=True,
    )
    value = output.strip().splitlines()
    if not value:
        return None
    candidate = value[-1].strip()
    try:
        return int(candidate)
    except ValueError:
        return None


def fetch_student_learning(session_ids: list[int]) -> str:
    if not session_ids:
        return ""
    id_list = ", ".join(str(sid) for sid in session_ids)
    return run_psql(
        "SELECT id, student_learning "
        "FROM sessions "
        f"WHERE id IN ({id_list}) "
        "ORDER BY id;"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    register_usage_callback()
    pass_headers = [
        "=== PASS 1: Learning Phase ===",
        "=== PASS 2: Applied Learning ===",
    ]

    metrics: list[PassMetrics] = []
    preview_learning = ""
    for index, header in enumerate(pass_headers):
        if index:
            print()
        pass_metrics = execute_pass(header)
        metrics.append(pass_metrics)
        if index == 0:
            preview_learning = wait_for_persona_learning(pass_metrics.persona_fingerprint)
            if preview_learning and "(0 rows)" not in preview_learning:
                print("\nPersona memory captured after Pass 1 (preview):")
                print(preview_learning)
            else:
                print("\n[warning] No persona memory persisted yet; continuing to Pass 2...")

    if len(metrics) == 2:
        print_efficiency(metrics[0], metrics[1])


if __name__ == "__main__":
    main()
