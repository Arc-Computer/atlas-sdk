"""Baseline runner using LiteLLM for direct model comparisons."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import litellm

SYSTEM_PROMPT = """You are a supply chain operations analyst specializing in inventory rebalancing decisions.

When presented with warehouse inventory scenarios, you must:
1. Analyze warehouse utilization, demand trends, and cost factors
2. Quantify economic trade-offs (transfer cost vs carrying cost vs stockout risk)
3. Consider operational constraints (capacity, timing, weather, supplier lead times)
4. Provide clear, actionable recommendations with specific calculations

Structure your analysis:
- Current Situation: Summarize warehouse states and demand
- Key Factors: Identify critical decision drivers
- Cost Analysis: Calculate transfer, carrying, and stockout costs
- Trade-offs: Explain conflicting priorities
- Recommendation: Clear action with quantified cost/benefit
- Risks: Identify key uncertainties

Always quantify your reasoning with specific numbers from the context.
"""


@dataclass
class UsageSnapshot:
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    reasoning_tokens: Optional[int]
    total_tokens: Optional[int]

    def summary(self) -> str:
        parts: List[str] = []
        if isinstance(self.prompt_tokens, (int, float)):
            parts.append(f"{int(self.prompt_tokens):,} prompt")
        if isinstance(self.completion_tokens, (int, float)):
            parts.append(f"{int(self.completion_tokens):,} completion")
        if isinstance(self.reasoning_tokens, (int, float)) and self.reasoning_tokens:
            parts.append(f"{int(self.reasoning_tokens):,} thinking")
        total = self.total_tokens
        if total is None and parts:
            total = (self.prompt_tokens or 0) + (self.completion_tokens or 0)
        out = " + ".join(parts)
        if total:
            if out:
                out += f" = {int(total):,} total"
            else:
                out = f"{int(total):,} total"
        return out or "0 tokens"

    def to_dict(self) -> Dict[str, int | None]:
        return {
            "prompt": int(self.prompt_tokens) if isinstance(self.prompt_tokens, (int, float)) else None,
            "completion": int(self.completion_tokens) if isinstance(self.completion_tokens, (int, float)) else None,
            "reasoning": int(self.reasoning_tokens) if isinstance(self.reasoning_tokens, (int, float)) else None,
            "total": int(self.total_tokens) if isinstance(self.total_tokens, (int, float)) else None,
        }


@dataclass
class BaselineRecord:
    scenario_id: str
    timestamp: str
    execution_time_seconds: float
    tokens: UsageSnapshot
    model: str
    answer: str
    status: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "timestamp": self.timestamp,
            "execution_time_seconds": self.execution_time_seconds,
            "tokens": self.tokens.to_dict(),
            "model": self.model,
            "answer": self.answer,
            "status": self.status,
        }


def load_scenarios(path: str) -> List[Dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    return payload if isinstance(payload, list) else [payload]


def build_messages(task: str, context: Dict[str, Any]) -> List[Dict[str, str]]:
    context_json = json.dumps(context, indent=2)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"{task}\n\nContext:\n{context_json}",
        },
    ]


def resolve_api_key(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key
    env_file = Path(".env")
    if env_file.exists():
        for raw_line in env_file.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("export "):
                line = line[len("export ") :]
            if "=" not in line:
                continue
            name, value = line.split("=", 1)
            if name.strip() == "OPENAI_API_KEY":
                cleaned = value.strip().strip("\"").strip("'")
                if cleaned:
                    os.environ.setdefault("OPENAI_API_KEY", cleaned)
                    return cleaned
    return None


def _response_to_dict(response: Any) -> Dict[str, Any]:
    if hasattr(response, "model_dump"):
        try:
            payload = response.model_dump()
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    if hasattr(response, "dict"):
        try:
            payload = response.dict()
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    if isinstance(response, dict):
        return response
    return {}


def extract_usage(response: Any) -> UsageSnapshot:
    payload = _response_to_dict(response)
    usage = payload.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    reasoning_tokens = None
    details = usage.get("completion_tokens_details")
    if isinstance(details, dict):
        reasoning_tokens = details.get("reasoning_tokens")
    return UsageSnapshot(prompt_tokens, completion_tokens, reasoning_tokens, total_tokens)


def extract_answer(response: Any) -> str:
    payload = _response_to_dict(response)
    choices = payload.get("choices") or []
    if not choices:
        return json.dumps(payload, indent=2)
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        fragments = [part.get("text") for part in content if isinstance(part, dict) and isinstance(part.get("text"), str)]
        if fragments:
            return "".join(fragments)
    return json.dumps(payload, indent=2)


def run_baseline(
    scenarios_path: str,
    output_dir: str,
    limit: Optional[int],
    model: str,
    api_key: Optional[str],
    reasoning_effort: Optional[str],
) -> None:
    scenarios = load_scenarios(scenarios_path)
    if limit:
        scenarios = scenarios[:limit]

    key = resolve_api_key(api_key)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    results: List[BaselineRecord] = []
    for index, scenario in enumerate(scenarios, start=1):
        scenario_id = scenario.get("scenario_id", f"scenario_{index}")
        task = scenario.get("task", "")
        context = scenario.get("context", {})

        messages = build_messages(task, context)
        start = time.time()

        print("\n" + "─" * 80)
        print(f"Running Baseline {index}/{len(scenarios)}: {scenario_id}")
        print("─" * 80 + "\n")

        try:
            litellm.drop_params = True
            response = litellm.completion(
                model=model,
                messages=messages,
                api_key=key,
                reasoning={"effort": reasoning_effort} if reasoning_effort else None,
                max_tokens=32000,
            )
            usage = extract_usage(response)
            answer_text = extract_answer(response)
            status = "success"
            duration = time.time() - start
            print(f"✓ Completed in {duration:.2f}s")
            print(f"  Tokens: {usage.summary()}\n")
        except Exception as exc:  # pragma: no cover
            usage = UsageSnapshot(None, None, None, None)
            answer_text = f"Baseline run failed: {exc}"
            status = "error"
            duration = time.time() - start
            print(f"✗ Baseline failed in {duration:.2f}s\n  Error: {exc}\n")

        record = BaselineRecord(
            scenario_id=scenario_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            execution_time_seconds=duration,
            tokens=usage,
            model=model,
            answer=answer_text,
            status=status,
        )
        results.append(record)

        per_run_path = output_root / f"baseline_{scenario_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        per_run_path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")

    summary_path = output_root / f"baseline_results_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.write_text(json.dumps([r.to_dict() for r in results], indent=2), encoding="utf-8")
    print(f"\n✓ Baseline results written to {summary_path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the baseline LLM using LiteLLM.")
    parser.add_argument("--scenarios", default="supply_chain/output/scenarios.json", help="Path to scenarios JSON.")
    parser.add_argument("--output", default="supply_chain/baseline_logs", help="Directory for baseline logs.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of scenarios to run.")
    parser.add_argument("--model", default="gpt-5-mini", help="LLM model identifier (default: %(default)s).")
    parser.add_argument("--api-key", dest="api_key", default=None, help="Optional OpenAI API key override.")
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default=None,
        help="Optional reasoning effort hint passed to LiteLLM/OpenAI.",
    )
    args = parser.parse_args()
    run_baseline(args.scenarios, args.output, args.limit, args.model, args.api_key, args.reasoning_effort)


if __name__ == "__main__":
    main()
