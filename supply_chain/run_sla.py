"""Run Atlas across a bundle of supply-chain scenarios and emit SLA-style logs."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from atlas import core
from atlas.runtime.orchestration.execution_context import ExecutionContext


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

SLA_LEARNING_KEY = "supply_chain_sla_shared_history"


def load_scenarios(path: str) -> List[Dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    return payload if isinstance(payload, list) else [payload]


async def _execute(task: str, config_path: str) -> Tuple[Any, Dict[str, Any]]:
    result = await core.arun(
        task=task,
        config_path=config_path,
        stream_progress=True,
        session_metadata={
            "learning_key": SLA_LEARNING_KEY,
            "tags": ["supply_chain_sla"],
        },
    )
    metadata = deepcopy(ExecutionContext.get().metadata)
    return result, metadata


def _summarise_learning(history: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(history, dict):
        return {}
    summary: Dict[str, Any] = {}
    count = history.get("count")
    if isinstance(count, (int, float)):
        summary["count"] = int(count)
    average = history.get("average_score")
    if isinstance(average, (int, float)):
        summary["average_score"] = float(average)
    if history.get("entries"):
        summary["has_entries"] = True
    return summary


def _token_totals(metadata: Dict[str, Any]) -> Dict[str, int]:
    usage = metadata.get("token_usage") if isinstance(metadata, dict) else None
    if not isinstance(usage, dict):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "reasoning_tokens": 0, "calls": 0}

    def _coerce(value: Any) -> int:
        if isinstance(value, (int, float)):
            return int(value)
        return 0

    prompt = _coerce(usage.get("prompt_tokens"))
    completion = _coerce(usage.get("completion_tokens"))
    total = _coerce(usage.get("total_tokens"))
    reasoning = _coerce(usage.get("reasoning_tokens"))
    if total == 0:
        total = prompt + completion
    if reasoning == 0:
        details = usage.get("completion_tokens_details")
        if isinstance(details, dict):
            reasoning = _coerce(details.get("reasoning_tokens"))
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "reasoning_tokens": reasoning,
        "calls": _coerce(usage.get("calls")),
    }


def extract_metrics(result: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
    session_reward = metadata.get("session_reward")
    learning_history = metadata.get("learning_history")
    token_usage = _token_totals(metadata)

    metrics = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "execution": {
            "mode": metadata.get("execution_mode", "unknown"),
            "confidence": metadata.get("adaptive_summary", {}).get("confidence") if isinstance(metadata.get("adaptive_summary"), dict) else None,
            "duration_seconds": None,
            "status": "success" if result.final_answer else "failed",
        },
        "reward": {
            "score": session_reward.get("score") if isinstance(session_reward, dict) else None,
            "uncertainty": session_reward.get("uncertainty") if isinstance(session_reward, dict) else None,
            "principles": session_reward.get("principles") if isinstance(session_reward, dict) else [],
            "student_learning": metadata.get("session_student_learning"),
            "teacher_learning": metadata.get("session_teacher_learning"),
        },
        "steps": {
            "total": len(result.step_results),
            "retries": sum(max(0, step.attempts - 1) for step in result.step_results),
            "details": [
                {
                    "step_id": step.step_id,
                    "description": step.metadata.get("structured_output", {}).get("step", {}).get("description") if isinstance(step.metadata, dict) else "",
                    "attempts": step.attempts,
                    "status": step.metadata.get("status") if isinstance(step.metadata, dict) else "completed",
                    "reward_score": getattr(step.evaluation.reward, "score", None),
                }
                for step in result.step_results
            ],
        },
        "tokens": token_usage,
        "learning": {
            "key": metadata.get("learning_key"),
            "history": _summarise_learning(learning_history),
        },
        "final_answer": result.final_answer,
        "plan": result.plan.model_dump() if result.plan else None,
    }

    if isinstance(session_reward, dict) and isinstance(session_reward.get("score"), (int, float)):
        score = float(session_reward["score"])
        summary = metrics["learning"]["history"]
        count = summary.get("count", 0)
        if count:
            total = summary.get("average_score", 0.0) * count + score
            summary["average_score"] = total / (count + 1)
            summary["count"] = count + 1
        else:
            summary.update({"average_score": score, "count": 1})

    return metrics


def print_summary(metrics: Dict[str, Any], scenario_id: str) -> None:
    print("\n" + "=" * 80)
    print("SLA EXECUTION SUMMARY")
    print("=" * 80)
    print(f"\nScenario ID: {scenario_id}")
    print(f"Timestamp: {metrics['timestamp']}")

    exec_meta = metrics["execution"]
    print("\n[EXECUTION]")
    print(f"  Mode: {exec_meta['mode'].upper()}")
    if exec_meta.get("confidence") is not None:
        print(f"  Confidence: {exec_meta['confidence']:.2f}")
    print(f"  Status: {exec_meta['status'].upper()}")
    if exec_meta.get("duration_seconds"):
        print(f"  Duration: {exec_meta['duration_seconds']:.2f}s")

    reward = metrics["reward"]
    print("\n[REWARD]")
    if isinstance(reward.get("score"), (int, float)):
        print(f"  Score: {reward['score']:.3f}")
        if isinstance(reward.get("uncertainty"), (int, float)):
            print(f"  Uncertainty: {reward['uncertainty']:.3f}")
        if reward.get("principles"):
            print(f"  Principles ({len(reward['principles'])}):")
            for principle in reward["principles"]:
                name = principle.get("name")
                weight = principle.get("weight")
                if name:
                    print(f"    - {name}: {float(weight):.2f}" if isinstance(weight, (int, float)) else f"    - {name}")
        if reward.get("student_learning"):
            snippet = reward["student_learning"][:200]
            print("  Student Learning:")
            print(f"    {snippet}...")
    else:
        print("  No reward evaluation available")

    learning = metrics["learning"]
    history = learning.get("history") or {}
    if learning.get("key") or history:
        print("\n[LEARNING]")
        if learning.get("key"):
            print(f"  Learning Key: {learning['key']}")
        count = history.get("count")
        avg = history.get("average_score")
        if isinstance(count, (int, float)):
            line = f"  History: {int(count)} entries"
            if isinstance(avg, (int, float)):
                line += f", avg score={avg:.2f}"
            print(line)
        else:
            print("  History: None recorded yet")

    steps = metrics["steps"]
    print("\n[STEPS]")
    print(f"  Total: {steps['total']}")
    print(f"  Retries: {steps['retries']}")
    for detail in steps["details"]:
        retry = f" ({detail['attempts']} attempts)" if detail["attempts"] > 1 else ""
        score = f" | score={detail['reward_score']:.2f}" if isinstance(detail["reward_score"], (int, float)) else ""
        print(f"    Step {detail['step_id']}: {detail['status']}{retry}{score}")

    tokens = metrics["tokens"]
    total_tokens = tokens.get("total_tokens") or (tokens.get("prompt_tokens", 0) + tokens.get("completion_tokens", 0))
    reasoning_tokens = tokens.get("reasoning_tokens", 0)
    if total_tokens or reasoning_tokens:
        print("\n[TOKENS]")
        print(f"  Prompt: {tokens.get('prompt_tokens', 0):,}")
        print(f"  Completion: {tokens.get('completion_tokens', 0):,}")
        print(f"  Total: {total_tokens:,}")
        if tokens.get("calls"):
            print(f"  LLM Calls: {tokens['calls']:,}")
        if reasoning_tokens:
            print(f"  Thinking: {reasoning_tokens:,}")

    print("\n" + "=" * 80 + "\n")


def run_sla(scenarios_path: str, config_path: str, output_dir: str, limit: int | None) -> None:
    scenarios = load_scenarios(scenarios_path)
    if limit:
        scenarios = scenarios[:limit]

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_records: List[Dict[str, Any]] = []

    for index, scenario in enumerate(scenarios, start=1):
        scenario_id = scenario.get("scenario_id", f"scenario_{index}")
        task = scenario.get("task", "")
        context = scenario.get("context", {})
        task_with_context = f"{task}\n\nContext:\n{json.dumps(context, indent=2)}"

        print("\n" + "─" * 80)
        print(f"Running Scenario {index}/{len(scenarios)}: {scenario_id}")
        print("─" * 80 + "\n")

        start = time.time()
        try:
            result, metadata = asyncio.run(_execute(task_with_context, config_path))
        except KeyboardInterrupt:  # pragma: no cover - manual interruption
            print("Run interrupted by user.", file=sys.stderr)
            raise
        except Exception as exc:
            print(f"✗ Scenario {scenario_id} failed: {exc}")
            duration = time.time() - start
            summary_records.append(
                {
                    "scenario_id": scenario_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "execution": {
                        "mode": "unknown",
                        "confidence": None,
                        "duration_seconds": duration,
                        "status": "error",
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                }
            )
            continue

        duration = time.time() - start
        metrics = extract_metrics(result, metadata)
        metrics["scenario_id"] = scenario_id
        metrics["execution"]["duration_seconds"] = duration

        print_summary(metrics, scenario_id)

        log_path = output_root / f"sla_log_{scenario_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        log_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
        summary_records.append(metrics)

    summary_path = output_root / f"sla_summary_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.write_text(json.dumps(summary_records, indent=2, default=str), encoding="utf-8")
    print(f"\n✓ Summary saved to {summary_path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Atlas agent across supply-chain scenarios.")
    parser.add_argument("--scenarios", default="supply_chain/output/scenarios.json", help="Path to scenarios JSON.")
    parser.add_argument("--config", default="supply_chain/atlas_config.yaml", help="Atlas config path.")
    parser.add_argument("--output", default="supply_chain/sla_logs", help="Directory for SLA logs.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of scenarios to run.")
    args = parser.parse_args()
    run_sla(args.scenarios, args.config, args.output, args.limit)


if __name__ == "__main__":
    main()
