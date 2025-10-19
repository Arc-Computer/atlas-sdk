from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from atlas.config.models import AdaptiveProbeConfig, LLMParameters, LLMProvider
from atlas.utils.env import load_dotenv_if_available
from atlas.runtime.adaptive import CapabilityProbeClient

DEFAULT_DATASET = Path("atlas/data/sample_probe_payloads.jsonl")
DEFAULT_MODELS = ("gemini", "anthropic", "xai")


MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "gemini": {
        "label": "Gemini 2.5 Flash",
        "model_env": "ATLAS_PROBE_MODEL_GEMINI",
        "default_model": "gemini/gemini-2.5-flash",
        "api_key_env": "GEMINI_API_KEY",
        "provider": LLMProvider.GOOGLE,
        "timeout": 20.0,
        "temperature": 0.2,
    },
    "anthropic": {
        "label": "Claude Haiku 4.5",
        "model_env": "ATLAS_PROBE_MODEL_ANTHROPIC",
        "default_model": "anthropic/claude-haiku-4-5",
        "api_key_env": "ANTHROPIC_API_KEY",
        "provider": LLMProvider.ANTHROPIC,
        "timeout": 20.0,
        "temperature": 0.2,
    },
    "xai": {
        "label": "Grok 4 Fast",
        "model_env": "ATLAS_PROBE_MODEL_XAI",
        "default_model": "xai/grok-4-fast",
        "api_key_env": "XAI_API_KEY",
        "provider": LLMProvider.XAI,
        "timeout": 20.0,
        "temperature": 0.2,
    },
}


@dataclass(slots=True)
class ProbeSample:
    task: str
    learning_history: dict[str, Any]
    expected_mode: str | None
    metadata: dict[str, Any]


@dataclass(slots=True)
class ProbeResult:
    sample: ProbeSample
    model: str
    mode: str | None
    confidence: float | None
    latency: float | None
    error: str | None


def load_dataset(path: Path) -> list[ProbeSample]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    samples: list[ProbeSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            samples.append(
                ProbeSample(
                    task=str(payload.get("task")),
                    learning_history=dict(payload.get("learning_history") or {}),
                    expected_mode=payload.get("expected_mode"),
                    metadata=dict(payload.get("metadata") or {}),
                )
            )
    if not samples:
        raise ValueError(f"Dataset {path} is empty.")
    return samples


def build_parameters(model_key: str) -> LLMParameters:
    preset = MODEL_PRESETS.get(model_key)
    if not preset:
        raise ValueError(f"Unknown model preset '{model_key}'. Supported: {', '.join(MODEL_PRESETS)}")

    model_name = (
        os.environ.get(preset["model_env"])
        or os.environ.get(f"{preset['model_env']}_MODEL")
        or preset["default_model"]
    )
    api_key_env = os.environ.get(f"{preset['model_env']}_API_KEY") or preset["api_key_env"]

    return LLMParameters(
        provider=preset["provider"],
        model=model_name,
        api_key_env=api_key_env,
        temperature=preset["temperature"],
        timeout_seconds=float(os.environ.get("ATLAS_PROBE_TIMEOUT", preset["timeout"])),
    )


async def evaluate_model(
    model_key: str,
    samples: Sequence[ProbeSample],
    repeats: int,
) -> list[ProbeResult]:
    parameters = build_parameters(model_key)
    client = CapabilityProbeClient(AdaptiveProbeConfig(llm=parameters))
    results: list[ProbeResult] = []
    for sample in samples:
        for _ in range(repeats):
            start = time.perf_counter()
            try:
                decision = await client.arun(
                    task=sample.task,
                    dossier={},
                    execution_metadata={"learning_history": sample.learning_history},
                )
                latency = time.perf_counter() - start
                results.append(
                    ProbeResult(
                        sample=sample,
                        model=model_key,
                        mode=decision.mode,
                        confidence=decision.confidence,
                        latency=latency,
                        error=None,
                    )
                )
            except Exception as exc:  # pragma: no cover - network errors
                results.append(
                    ProbeResult(
                        sample=sample,
                        model=model_key,
                        mode=None,
                        confidence=None,
                        latency=None,
                        error=str(exc),
                    )
                )
    return results


def summarise_results(results: Sequence[ProbeResult]) -> dict[str, Any]:
    total = len(results)
    successes = [res for res in results if res.error is None]
    failures = total - len(successes)
    latencies = [res.latency for res in successes if res.latency is not None]
    accuracy = None
    expected = [res for res in successes if res.sample.expected_mode]
    if expected:
        correct = sum(1 for res in expected if res.mode == res.sample.expected_mode)
        accuracy = correct / len(expected)
    mode_counts = Counter(res.mode for res in successes if res.mode)

    latency_summary = None
    if latencies:
        latency_summary = {
            "mean": statistics.mean(latencies),
            "median": statistics.median(latencies),
            "p95": percentile(latencies, 95),
        }

    return {
        "total": total,
        "success": len(successes),
        "failure": failures,
        "accuracy": accuracy,
        "mode_counts": dict(mode_counts),
        "latency_seconds": latency_summary,
    }


def percentile(values: Sequence[float], percentile_rank: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * percentile_rank / 100
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    if f == c:
        return sorted_values[int(k)]
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return d0 + d1


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate capability probe models across a dataset of tasks + learning histories.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"JSONL dataset file (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help=f"Model presets to evaluate (default: {', '.join(DEFAULT_MODELS)}).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of times to evaluate each sample per model (default: 1).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write JSON results.",
    )
    return parser.parse_args(argv)


async def main_async(argv: Sequence[str] | None = None) -> int:
    load_dotenv_if_available()
    args = parse_args(argv)
    samples = load_dataset(args.dataset)

    all_results: dict[str, list[ProbeResult]] = {}
    for model_key in args.models:
        print(f"Evaluating {model_key}...", file=sys.stderr)
        results = await evaluate_model(model_key, samples, max(args.repeats, 1))
        all_results[model_key] = results

    summaries = {model: summarise_results(results) for model, results in all_results.items()}
    print("\n=== Capability Probe Evaluation Summary ===")
    for model_key, summary in summaries.items():
        preset = MODEL_PRESETS.get(model_key, {})
        label = preset.get("label", model_key)
        print(f"\nModel: {label} ({model_key})")
        for key, value in summary.items():
            print(f"  {key}: {value}")

    if args.output:
        payload = {
            "summaries": summaries,
            "results": {
                model: [
                    {
                        "task": result.sample.task,
                        "mode": result.mode,
                        "confidence": result.confidence,
                        "latency": result.latency,
                        "error": result.error,
                        "expected_mode": result.sample.expected_mode,
                        "metadata": result.sample.metadata,
                    }
                    for result in results
                ]
                for model, results in all_results.items()
            },
        }
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nDetailed results written to {args.output}", file=sys.stderr)

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return asyncio.run(main_async(argv))


if __name__ == "__main__":
    raise SystemExit(main())
