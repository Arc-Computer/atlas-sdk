from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from atlas.config.loader import load_config
from atlas.config.models import LLMParameters, LLMProvider, RIMConfig
from atlas.evaluation import Evaluator, SessionStepRecord, SessionTrajectory
from atlas.evaluation.evaluator import SessionSample
from atlas.runtime.orchestration.execution_context import ExecutionContext
from atlas.types import Plan, Step
from atlas.utils.env import load_dotenv_if_available

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    yaml = None

DEFAULT_DATASET = Path("atlas/data/reward_eval_trajectories.jsonl")
DEFAULT_CONFIG = Path("configs/examples/openai_agent.yaml")
REWARD_CONFIG_PATH = Path("configs/eval/reward_system.yaml")


@dataclass(frozen=True)
class TrajectoryRecord:
    identifier: str
    trajectory: SessionTrajectory
    task: str
    adaptive_summary: dict[str, Any]
    session_metadata: dict[str, Any] | None
    trajectory_type: str | None


@dataclass(frozen=True)
class JudgeCombo:
    identifier: str
    small_preset: str
    large_preset: str
    description: str


DEFAULT_JUDGE_PRESETS: dict[str, dict[str, Any]] = {
    "gemini-2.5-flash": {
        "provider": LLMProvider.GEMINI,
        "model": "gemini/gemini-2.5-flash",
        "api_key_env": "GEMINI_API_KEY",
        "temperature": 0.2,
        "timeout_seconds": 60.0,
        "max_output_tokens": 8096,
    },
    "gemini-2.5-pro": {
        "provider": LLMProvider.GEMINI,
        "model": "gemini/gemini-2.5-pro",
        "api_key_env": "GEMINI_API_KEY",
        "temperature": 0.2,
        "timeout_seconds": 60.0,
        "max_output_tokens": 8096,
    },
    "claude-haiku-4-5": {
        "provider": LLMProvider.ANTHROPIC,
        "model": "claude-haiku-4-5",
        "api_key_env": "ANTHROPIC_API_KEY",
        "temperature": 0.2,
        "timeout_seconds": 60.0,
    },
    "claude-sonnet-4-5-20250929": {
        "provider": LLMProvider.ANTHROPIC,
        "model": "claude-sonnet-4-5-20250929",
        "api_key_env": "ANTHROPIC_API_KEY",
        "temperature": 0.15,
        "timeout_seconds": 60.0,
    },
    "grok-4-fast": {
        "provider": LLMProvider.XAI,
        "model": "xai/grok-4-fast",
        "api_key_env": "XAI_API_KEY",
        "temperature": 0.2,
        "timeout_seconds": 45.0,
    },
    "grok-4": {
        "provider": LLMProvider.XAI,
        "model": "xai/grok-4",
        "api_key_env": "XAI_API_KEY",
        "temperature": 0.2,
        "timeout_seconds": 60.0,
    },
    "gpt-5-mini": {
        "provider": LLMProvider.OPENAI,
        "model": "gpt-5-mini",
        "api_key_env": "OPENAI_API_KEY",
        "temperature": 0.2,
        "timeout_seconds": 60.0,
    },
    "gpt-5": {
        "provider": LLMProvider.OPENAI,
        "model": "gpt-5",
        "api_key_env": "OPENAI_API_KEY",
        "temperature": 0.2,
        "timeout_seconds": 60.0,
    },
}


DEFAULT_JUDGE_COMBOS: dict[str, JudgeCombo] = {
    "gemini_pair": JudgeCombo(
        identifier="gemini_pair",
        small_preset="gemini-2.5-flash",
        large_preset="gemini-2.5-pro",
        description="Gemini Flash (small) ➜ Gemini Pro (arbiter)",
    ),
    "claude_stack": JudgeCombo(
        identifier="claude_stack",
        small_preset="claude-haiku-4-5",
        large_preset="claude-sonnet-4-5-20250929",
        description="Claude Haiku ➜ Claude Sonnet",
    ),
    "gpt5_stack": JudgeCombo(
        identifier="gpt5_stack",
        small_preset="gpt-5-mini",
        large_preset="gpt-5",
        description="GPT-5 Mini ➜ GPT-5",
    ),
    "grok_stack": JudgeCombo(
        identifier="grok_stack",
        small_preset="grok-4-fast",
        large_preset="grok-4",
        description="Grok 4 Fast ➜ Grok 4",
    ),
}


def _coerce_provider(raw: Any) -> LLMProvider:
    if isinstance(raw, LLMProvider):
        return raw
    if isinstance(raw, str):
        try:
            return LLMProvider[raw]
        except KeyError:
            raise ValueError(f"Unknown provider value '{raw}'") from None
    raise ValueError(f"Provider must be LLMProvider or string, got {type(raw)!r}")


def _load_reward_system_config(
    path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, JudgeCombo]]:
    presets = {name: preset.copy() for name, preset in DEFAULT_JUDGE_PRESETS.items()}
    combos = DEFAULT_JUDGE_COMBOS.copy()

    if not path.exists():
        return presets, combos

    if yaml is None:
        print(
            f"[reward-eval] PyYAML not installed; ignoring {path}. Install pyyaml to customize reward presets.",
            file=sys.stderr,
        )
        return presets, combos

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except Exception as exc:  # pragma: no cover - configuration parsing
        print(f"[reward-eval] Failed to parse {path}: {exc}", file=sys.stderr)
        return presets, combos

    raw_presets = payload.get("presets", {}) if isinstance(payload, dict) else {}
    if isinstance(raw_presets, dict):
        for name, data in raw_presets.items():
            if not isinstance(data, dict):
                continue
            if name not in presets and "provider" not in data:
                print(
                    f"[reward-eval] Skipping preset '{name}': missing required 'provider' field for new preset.",
                    file=sys.stderr,
                )
                continue
            try:
                provider = _coerce_provider(data.get("provider", presets.get(name, {}).get("provider")))
            except ValueError as exc:
                print(f"[reward-eval] Skipping preset '{name}': {exc}", file=sys.stderr)
                continue
            preset = {
                "provider": provider,
                "model": data.get("model"),
                "api_key_env": data.get("api_key_env"),
                "temperature": data.get("temperature", presets.get(name, {}).get("temperature", 0.2)),
                "timeout_seconds": data.get(
                    "timeout_seconds",
                    presets.get(name, {}).get("timeout_seconds"),
                ),
                "max_output_tokens": data.get(
                    "max_output_tokens",
                    presets.get(name, {}).get("max_output_tokens"),
                ),
            }
            presets[name] = {key: value for key, value in preset.items() if value is not None}

    raw_combos = payload.get("combos", {}) if isinstance(payload, dict) else {}
    if isinstance(raw_combos, dict):
        for name, data in raw_combos.items():
            if not isinstance(data, dict):
                continue
            small = data.get("small_preset")
            large = data.get("large_preset")
            if not small or not large:
                print(f"[reward-eval] Combo '{name}' missing small/large presets; skipping.", file=sys.stderr)
                continue
            description = data.get("description") or (
                DEFAULT_JUDGE_COMBOS[name].description if name in DEFAULT_JUDGE_COMBOS else name
            )
            combos[name] = JudgeCombo(
                identifier=name,
                small_preset=small,
                large_preset=large,
                description=description,
            )

    return presets, combos


JUDGE_PRESETS, JUDGE_COMBOS = _load_reward_system_config(REWARD_CONFIG_PATH)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, set):
        try:
            ordered = sorted(value)  # type: ignore[call-arg]
        except TypeError:
            ordered = sorted((repr(item), item) for item in value)
            return [_json_safe(item) for _, item in ordered]
        return [_json_safe(item) for item in ordered]
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    for attr in ("model_dump", "dict"):
        if hasattr(value, attr):
            try:
                dumped = getattr(value, attr)()
            except Exception:
                continue
            return _json_safe(dumped)
    return repr(value)


class HarnessEvaluator(Evaluator):
    """Evaluator variant with relaxed payload parsing for offline reward sweeps."""

    def __init__(self, config: RIMConfig, *, collect_audit: bool = False) -> None:
        super().__init__(config)
        self._collect_audit = collect_audit

    async def _sample_session(  # type: ignore[override]
        self,
        trajectory: SessionTrajectory,
        focus_prompt: str | None,
        temperature: float,
    ) -> tuple[SessionSample | None, dict[str, Any] | None]:
        messages = self._build_session_messages(trajectory, focus_prompt)
        exec_context = ExecutionContext.get()
        exec_context.metadata["active_actor"] = "reward"
        exec_context.metadata["_reasoning_origin"] = ("reward", "sample")
        audit_entry: dict[str, Any] | None = None
        if self._collect_audit:
            audit_entry = {
                "stage": "tier1",
                "model": self._small_client.model,
                "temperature": temperature,
                "messages": messages,
            }
        try:
            response = await self._small_client.acomplete(
                messages,
                response_format={"type": "json_object"},
                overrides={"temperature": temperature},
            )
            if self._collect_audit and audit_entry is not None:
                audit_entry["response"] = response.content
                audit_entry["raw_response"] = response.raw
                if response.reasoning:
                    audit_entry["reasoning"] = response.reasoning
        except Exception as exc:
            exec_context.metadata.setdefault("_llm_reasoning_queue", [])
            exec_context.metadata["_llm_reasoning_queue"].clear()
            if self._collect_audit:
                if audit_entry is None:
                    audit_entry = {
                        "stage": "tier1",
                        "model": self._small_client.model,
                        "temperature": temperature,
                        "messages": messages,
                    }
                audit_entry["error"] = repr(exc)
            return (None, audit_entry) if self._collect_audit else (None, None)

        payload = self._try_parse_json(response.content)
        if payload is None:
            payload = self._extract_from_tool_calls(response.raw)
        payload = self._normalise_payload(payload)
        reasoning_queue = self._consume_reasoning_metadata("reward", "sample")
        if self._collect_audit and audit_entry is not None and reasoning_queue:
            audit_entry["reasoning_queue"] = reasoning_queue

        if payload is None:
            if self._collect_audit and audit_entry is not None:
                audit_entry["parse_error"] = "empty_or_unrecognised_payload"
            return (None, audit_entry) if self._collect_audit else (None, None)
        parsed = self._parse_session_payload(payload)
        if parsed is None:
            if self._collect_audit and audit_entry is not None:
                audit_entry["parse_error"] = "invalid_payload_shape"
            return (None, audit_entry) if self._collect_audit else (None, None)

        return SessionSample(
            score=parsed["score"],
            uncertainty=parsed["uncertainty"],
            rationale=parsed["rationale"],
            principles=parsed["principles"],
        ), audit_entry if self._collect_audit else None

    def _normalise_payload(self, payload: Any) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            return None
        if payload.keys() == {"evaluation_input"}:
            return None
        for key in ("evaluation_output", "output", "result", "reward", "payload"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                return self._normalise_payload(nested)
        if "score" in payload and "uncertainty" in payload:
            return payload
        return None

    def _extract_from_tool_calls(self, raw: Any) -> dict[str, Any] | None:
        raw_payload: dict[str, Any] | None = None
        if hasattr(raw, "model_dump"):
            try:
                raw_payload = raw.model_dump()  # type: ignore[attr-defined]
            except Exception:
                raw_payload = None
        elif hasattr(raw, "dict"):
            try:
                raw_payload = raw.dict()  # type: ignore[attr-defined]
            except Exception:
                raw_payload = None
        elif isinstance(raw, dict):
            raw_payload = raw
        if not raw_payload:
            return None
        choices = raw_payload.get("choices") or []
        for choice in choices:
            message = (
                choice.get("message")
                if isinstance(choice, dict)
                else getattr(choice, "message", None)
            )
            if message is None:
                continue
            tool_calls = (
                message.get("tool_calls")
                if isinstance(message, dict)
                else getattr(message, "tool_calls", None)
            )
            if not tool_calls:
                continue
            for call in tool_calls:
                function = (
                    call.get("function")
                    if isinstance(call, dict)
                    else getattr(call, "function", None)
                )
                if not function:
                    continue
                arguments = (
                    function.get("arguments")
                    if isinstance(function, dict)
                    else getattr(function, "arguments", None)
                )
                if not arguments:
                    continue
                try:
                    candidate = json.loads(arguments)
                except (TypeError, json.JSONDecodeError):
                    continue
                normalised = self._normalise_payload(candidate)
                if normalised is not None:
                    return normalised
        return None


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate reward judge pairings against captured session trajectories."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to reward evaluation JSONL dataset.",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Base Atlas config used to derive reward system settings.",
    )
    parser.add_argument(
        "--judge-combos",
        nargs="+",
        default=list(JUDGE_COMBOS),
        help=f"Judge combo identifiers to evaluate (default: {', '.join(JUDGE_COMBOS)}).",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="gemini_pair",
        help="Judge combo identifier to treat as baseline for deltas/correlation.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of times to replay each trajectory per judge combo.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum concurrent reward evaluations per combo.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write JSON report (per run details + summaries).",
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help="Optional path to write Markdown summary (defaults to results/reward/<timestamp>.md).",
    )
    parser.add_argument(
        "--collect-audit",
        action="store_true",
        help="Capture minimal model prompts/responses for debugging reward judge behaviour.",
    )
    return parser.parse_args(argv)


def load_reward_dataset(path: Path) -> list[TrajectoryRecord]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    records: list[TrajectoryRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            payload = _parse_dataset_line(line, line_number)
            identifier = f"trajectory_{len(records) + 1:03d}"
            plan_payload = payload["plan"]
            plan_model = Plan.model_validate(plan_payload)
            step_records = _build_step_records(payload["steps"])
            trajectory = SessionTrajectory(
                task=payload["task"],
                final_answer=payload["final_answer"],
                plan=plan_model.model_dump(),
                steps=step_records,
                execution_mode=payload.get("execution_mode"),
                teacher_intervened=bool(payload.get("teacher_intervened", False)),
                session_metadata=payload.get("session_metadata"),
                focus_prompt=payload.get("focus_prompt"),
            )
            record = TrajectoryRecord(
                identifier=identifier,
                trajectory=trajectory,
                task=payload["task"],
                adaptive_summary=payload.get("adaptive_summary") or {},
                session_metadata=payload.get("session_metadata"),
                trajectory_type=payload.get("trajectory_type"),
            )
            records.append(record)
    if not records:
        raise ValueError(f"Dataset {path} contained no trajectory records.")
    return records


def _parse_dataset_line(line: str, line_number: int) -> dict[str, Any]:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
    required_fields = ("task", "final_answer", "plan", "steps")
    for field in required_fields:
        if field not in payload:
            raise ValueError(f"Dataset line {line_number} missing required field '{field}'.")
    if not isinstance(payload["plan"], dict):
        raise ValueError(f"Dataset line {line_number} has invalid plan payload; expected object.")
    if not isinstance(payload["steps"], list) or not payload["steps"]:
        raise ValueError(f"Dataset line {line_number} must include a non-empty 'steps' array.")
    return payload


def _build_step_records(steps_payload: Sequence[dict[str, Any]]) -> list[SessionStepRecord]:
    records: list[SessionStepRecord] = []
    for index, entry in enumerate(steps_payload, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Step entry #{index} is not an object.")
        step_payload = entry.get("step")
        if not isinstance(step_payload, dict):
            raise ValueError(f"Step entry #{index} missing 'step' object.")
        step = Step.model_validate(step_payload)
        guidance_payload = entry.get("guidance") or []
        if guidance_payload and not isinstance(guidance_payload, list):
            raise ValueError(f"Step entry #{index} guidance must be an array of strings.")
        guidance = [str(item) for item in guidance_payload] if guidance_payload else None
        prior = entry.get("prior_results")
        if isinstance(prior, dict):

            normalised_prior = {}
            for key, value in prior.items():
                try:
                    cast_key = int(key)
                except (TypeError, ValueError):
                    cast_key = key
                normalised_prior[cast_key] = value
        elif prior is None:
            normalised_prior = None
        else:
            raise ValueError(f"Step entry #{index} prior_results must be an object or null.")
        validation = entry.get("validation")
        if validation is not None and not isinstance(validation, dict):
            raise ValueError(f"Step entry #{index} validation must be an object or null.")
        metadata = entry.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError(f"Step entry #{index} metadata must be an object or null.")
        record = SessionStepRecord(
            step=step,
            trace=str(entry.get("trace", "")),
            output=str(entry.get("output", "")),
            attempts=int(entry.get("attempts", 1)),
            guidance=guidance,
            status=str(entry.get("status")) if entry.get("status") is not None else None,
            validation=validation,
            prior_results=normalised_prior,
            metadata=metadata,
        )
        records.append(record)
    return records


def _build_llm_parameters(base: LLMParameters, preset_key: str) -> LLMParameters:
    preset = JUDGE_PRESETS.get(preset_key)
    if preset is None:
        supported = ", ".join(sorted(JUDGE_PRESETS))
        raise ValueError(f"Unknown judge preset '{preset_key}'. Supported options: {supported}")
    return base.model_copy(update=preset)


def _build_rim_config(base: RIMConfig, combo: JudgeCombo) -> RIMConfig:
    small = _build_llm_parameters(base.small_model, combo.small_preset)
    large = _build_llm_parameters(base.large_model, combo.large_preset)
    return base.model_copy(update={"small_model": small, "large_model": large})


EvaluatorFactory = Callable[[RIMConfig, bool], Evaluator]


async def evaluate_combo(
    combo: JudgeCombo,
    rim_config: RIMConfig,
    trajectories: Sequence[TrajectoryRecord],
    *,
    repeats: int,
    concurrency: int,
    evaluator_factory: EvaluatorFactory,
    collect_audit: bool,
) -> list[dict[str, Any]]:
    evaluator = evaluator_factory(rim_config, collect_audit)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    results: list[dict[str, Any]] = []

    async def _evaluate_single(record: TrajectoryRecord, repeat_index: int) -> dict[str, Any]:
        async with semaphore:
            context = ExecutionContext.get()
            context.reset()
            print(
                f"  ↳ Combo '{combo.identifier}': trajectory {record.identifier} "
                f"(repeat {repeat_index + 1}/{repeats})"
            )
            if record.session_metadata:
                context.metadata["session_metadata"] = record.session_metadata
            adaptive_summary = record.adaptive_summary or {}
            if adaptive_summary:
                context.metadata["adaptive_summary"] = adaptive_summary
                adaptive_meta = {
                    "active_mode": adaptive_summary.get("adaptive_mode"),
                    "mode_history": list(adaptive_summary.get("mode_history", [])),
                }
                probe = adaptive_summary.get("probe")
                if probe:
                    adaptive_meta["probe"] = probe
                context.metadata["adaptive"] = adaptive_meta
            start = time.perf_counter()
            error: str | None = None
            evaluation = None
            try:
                evaluation = await evaluator.aevaluate_session(record.trajectory)
            except Exception as exc:  # pragma: no cover - network/LLM errors
                error = repr(exc)
            latency = time.perf_counter() - start
            context.reset()
            score: float | None = None
            uncertainties: list[float] = []
            escalated: bool | None = None
            samples: list[dict[str, Any]] | None = None
            audit_entries: list[dict[str, Any]] | None = None
            if evaluation is not None and error is None:
                score = float(evaluation.reward.score)
                samples_payload = evaluation.reward.raw.get("samples") if isinstance(evaluation.reward.raw, dict) else None
                if isinstance(samples_payload, list):
                    samples = samples_payload
                    for sample_entry in samples_payload:
                        uncertainty = sample_entry.get("uncertainty")
                        if isinstance(uncertainty, (int, float)):
                            uncertainties.append(float(uncertainty))
                escalated = any(judge.escalated for judge in evaluation.reward.judges)
                if collect_audit and evaluation.audit:
                    audit_entries = [_json_safe(entry) for entry in evaluation.audit]
            uncertainty_value = statistics.fmean(uncertainties) if uncertainties else None
            return {
                "combo": combo.identifier,
                "repeat_index": repeat_index,
                "trajectory_id": record.identifier,
                "task": record.task,
                "trajectory_type": record.trajectory_type,
                "score": score,
                "uncertainty": uncertainty_value,
                "escalated": escalated,
                "latency_ms": latency * 1000.0,
                "error": error,
                "samples": samples,
                "audit": audit_entries,
            }

    for repeat_index in range(repeats):
        for record in trajectories:
            results.append(await _evaluate_single(record, repeat_index))
    return results


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * percentile
    lower = int(k)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = k - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * weight


def _compute_correlation(pairs: Sequence[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    std_x = statistics.pstdev(xs)
    std_y = statistics.pstdev(ys)
    if std_x == 0.0 or std_y == 0.0:
        return None
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(ys)
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in pairs) / len(pairs)
    return covariance / (std_x * std_y)


def aggregate_results(
    per_run: Sequence[dict[str, Any]],
    *,
    baseline_combo: str | None,
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    per_combo: dict[str, list[dict[str, Any]]] = {}
    for record in per_run:
        per_combo.setdefault(record["combo"], []).append(record)

    baseline_scores: dict[tuple[str, int], float] = {}
    if baseline_combo:
        for record in per_combo.get(baseline_combo, []):
            if record["score"] is not None:
                key = (record["trajectory_id"], int(record["repeat_index"]))
                baseline_scores[key] = float(record["score"])

    for combo_id, records in per_combo.items():
        scores = [float(r["score"]) for r in records if r["score"] is not None]
        uncertainties = [float(r["uncertainty"]) for r in records if r["uncertainty"] is not None]
        latencies = [float(r["latency_ms"]) for r in records if r["score"] is not None]
        escalations = [bool(r["escalated"]) for r in records if r["escalated"] is not None]
        failures = [r for r in records if r["score"] is None]

        baseline_pairs: list[tuple[float, float]] = []
        deltas: list[float] = []
        within_threshold = 0
        total_with_baseline = 0
        for record in records:
            key = (record["trajectory_id"], int(record["repeat_index"]))
            baseline_score = baseline_scores.get(key)
            if record["score"] is not None and baseline_score is not None:
                total_with_baseline += 1
                delta = float(record["score"]) - baseline_score
                record["score_delta"] = delta
                deltas.append(delta)
                baseline_pairs.append((baseline_score, float(record["score"])))
                if abs(delta) <= 0.02:
                    within_threshold += 1
            else:
                record["score_delta"] = None

        agreement_payload = None
        if total_with_baseline:
            correlation = _compute_correlation(baseline_pairs)
            agreement_payload = {
                "mean_delta": statistics.fmean(deltas) if deltas else 0.0,
                "median_delta": statistics.median(deltas) if deltas else 0.0,
                "pearson": correlation,
                "within_0_02": within_threshold / total_with_baseline,
                "samples": total_with_baseline,
            }

        summaries[combo_id] = {
            "combo": combo_id,
            "runs": len(records),
            "successes": len(scores),
            "failures": len(failures),
            "score_mean": statistics.fmean(scores) if scores else None,
            "score_stdev": statistics.pstdev(scores) if len(scores) > 1 else 0.0 if scores else None,
            "uncertainty_mean": statistics.fmean(uncertainties) if uncertainties else None,
            "uncertainty_stdev": statistics.pstdev(uncertainties) if len(uncertainties) > 1 else 0.0 if uncertainties else None,
            "escalation_rate": (sum(1 for flag in escalations if flag) / len(escalations)) if escalations else None,
            "latency_mean_ms": statistics.fmean(latencies) if latencies else None,
            "latency_median_ms": statistics.median(latencies) if latencies else None,
            "latency_p95_ms": _percentile(latencies, 0.95),
            "agreement": agreement_payload,
        }
    return summaries


def render_summary_table(
    summaries: dict[str, dict[str, Any]],
    combos: dict[str, JudgeCombo],
    *,
    baseline: str | None,
) -> None:
    headers = [
        "Combo",
        "Score (avg)",
        "Score σ",
        "Escalation",
        "Uncertainty",
        "Latency ms",
        "Failures",
    ]
    lines = []
    for combo_id, summary in summaries.items():
        combo = combos.get(combo_id)
        label = combo.description if combo else combo_id
        if combo_id == baseline:
            label = f"{label} [baseline]"
        score_mean = summary["score_mean"]
        score_std = summary["score_stdev"]
        escalation = summary["escalation_rate"]
        uncertainty = summary["uncertainty_mean"]
        latency = summary["latency_mean_ms"]
        failures = summary["failures"]
        lines.append(
            [
                label,
                f"{score_mean:.3f}" if score_mean is not None else "n/a",
                f"{score_std:.3f}" if score_std is not None else "n/a",
                f"{escalation:.2%}" if escalation is not None else "n/a",
                f"{uncertainty:.3f}" if uncertainty is not None else "n/a",
                f"{latency:.1f}" if latency is not None else "n/a",
                str(failures),
            ]
        )

    widths = [max(len(headers[idx]), *(len(row[idx]) for row in lines)) for idx in range(len(headers))]
    header_row = " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers)))
    separator = "-+-".join("-" * widths[idx] for idx in range(len(headers)))
    print(header_row)
    print(separator)
    for row in lines:
        print(" | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))))


def generate_markdown_report(
    summaries: dict[str, dict[str, Any]],
    combos: dict[str, JudgeCombo],
    *,
    metadata: dict[str, Any],
    baseline: str | None,
) -> str:
    lines: list[str] = []
    title = metadata.get("title") or "Reward Model Evaluation"
    lines.append(f"# {title}")
    generated = metadata.get("generated_at")
    if generated:
        lines.append(f"_Generated: {generated}_")
    lines.append("")
    dataset = metadata.get("dataset")
    if dataset:
        lines.append(f"- **Dataset:** `{dataset}`")
    baseline_label = baseline
    if baseline and baseline in combos:
        baseline_label = combos[baseline].description
    if baseline:
        lines.append(f"- **Baseline:** {baseline} ({baseline_label})")
    repeats = metadata.get("repeats")
    if repeats:
        lines.append(f"- **Repeats:** {repeats}")
    lines.append("")

    headers = [
        "Combo",
        "Score (avg)",
        "Score σ",
        "Escalation",
        "Uncertainty",
        "Latency ms",
        "Failures",
    ]
    lines.append(" | ".join(headers))
    lines.append(" | ".join("---" for _ in headers))

    ordered_combos = metadata.get("combos") or sorted(summaries)
    for combo_id in ordered_combos:
        summary = summaries.get(combo_id)
        if not summary:
            continue
        combo = combos.get(combo_id)
        label = combo.description if combo else combo_id
        if combo_id == baseline:
            label = f"{label} (baseline)"
        score_mean = summary.get("score_mean")
        score_std = summary.get("score_stdev")
        escalation = summary.get("escalation_rate")
        uncertainty = summary.get("uncertainty_mean")
        latency = summary.get("latency_mean_ms")
        failures = summary.get("failures")
        row = [
            label,
            f"{score_mean:.3f}" if score_mean is not None else "n/a",
            f"{score_std:.3f}" if score_std is not None else "n/a",
            f"{escalation:.2%}" if escalation is not None else "n/a",
            f"{uncertainty:.3f}" if uncertainty is not None else "n/a",
            f"{latency:.1f}" if latency is not None else "n/a",
            str(failures),
        ]
        lines.append(" | ".join(row))

    return "\n".join(lines) + "\n"


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv_if_available()
    config = load_config(str(args.base_config))
    dataset = load_reward_dataset(args.dataset)
    requested_combos = []
    for combo_id in args.judge_combos:
        combo = JUDGE_COMBOS.get(combo_id)
        if combo is None:
            supported = ", ".join(sorted(JUDGE_COMBOS))
            raise ValueError(f"Unknown judge combo '{combo_id}'. Supported: {supported}")
        requested_combos.append(combo)

    # Ensure baseline is evaluated first so deltas are available for subsequent combos.
    baseline_combo: JudgeCombo | None = None
    if args.baseline:
        baseline_combo = JUDGE_COMBOS.get(args.baseline)
        if baseline_combo is None:
            supported = ", ".join(sorted(JUDGE_COMBOS))
            raise ValueError(f"Unknown baseline combo '{args.baseline}'. Supported: {supported}")
        ordered_combos: list[JudgeCombo] = []
        if baseline_combo in requested_combos:
            ordered_combos.append(baseline_combo)
        for combo in requested_combos:
            if combo is baseline_combo:
                continue
            ordered_combos.append(combo)
        requested_combos = ordered_combos

    rim_base: RIMConfig = config.rim

    def evaluator_factory(rim_cfg: RIMConfig, collect_audit: bool) -> Evaluator:
        return HarnessEvaluator(rim_cfg, collect_audit=collect_audit)

    per_run_records: list[dict[str, Any]] = []
    for combo in requested_combos:
        print(f"Starting combo '{combo.identifier}' ({combo.description}) with {len(dataset)} trajectories × {max(1, args.repeats)} repeats...")
        rim_config = _build_rim_config(rim_base, combo)
        combo_records = await evaluate_combo(
            combo,
            rim_config,
            dataset,
            repeats=max(1, args.repeats),
            concurrency=max(1, args.concurrency),
            evaluator_factory=evaluator_factory,
            collect_audit=args.collect_audit,
        )
        per_run_records.extend(combo_records)
        print(f"Completed combo '{combo.identifier}'.")

    summaries = aggregate_results(
        per_run_records,
        baseline_combo=baseline_combo.identifier if baseline_combo else None,
    )
    render_summary_table(
        summaries,
        JUDGE_COMBOS,
        baseline=baseline_combo.identifier if baseline_combo else None,
    )

    generated_at = datetime.now(timezone.utc)
    timestamp_slug = generated_at.strftime("%Y%m%dT%H%M%SZ")
    report = {
        "metadata": {
            "title": "Reward Model Evaluation",
            "dataset": str(args.dataset),
            "base_config": str(args.base_config),
            "combos": [combo.identifier for combo in requested_combos],
            "baseline": baseline_combo.identifier if baseline_combo else None,
            "repeats": max(1, args.repeats),
            "concurrency": max(1, args.concurrency),
            "generated_at": generated_at.isoformat(),
        },
        "per_run": per_run_records,
        "summaries": summaries,
    }

    markdown_path = args.markdown_output or Path("results/reward") / f"reward_eval_{timestamp_slug}.md"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_payload = generate_markdown_report(
        summaries,
        JUDGE_COMBOS,
        metadata=report["metadata"],
        baseline=report["metadata"]["baseline"],
    )
    markdown_path.write_text(markdown_payload, encoding="utf-8")
    print(f"Saved Markdown summary to {markdown_path}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"Wrote JSON report to {args.output}")
    return report


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:  # pragma: no cover - CLI convenience
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
