from __future__ import annotations

import json
import math
import logging
from statistics import fmean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from atlas.config.models import LLMParameters, LLMProvider, MetadataDigestConfig

DEFAULT_CHAR_BUDGET = 20000
MIN_CHAR_BUDGET = 5000
AVERAGE_CHARS_PER_TOKEN = 4
DEFAULT_CONTEXT_RATIO = 0.025
PROVIDER_TOKEN_LIMITS: Dict[LLMProvider, int] = {
    LLMProvider.ANTHROPIC: 200_000,
    LLMProvider.OPENAI: 128_000,
    LLMProvider.AZURE_OPENAI: 128_000,
    LLMProvider.BEDROCK: 200_000,
    LLMProvider.GEMINI: 1_000_000,
    LLMProvider.GOOGLE: 120_000,
    LLMProvider.XAI: 128_000,
}


class PromptDigestTooLargeError(RuntimeError):
    """Raised when a digest cannot be reduced below the provider budget."""


logger = logging.getLogger(__name__)


def build_prompt_digest(
    metadata: Mapping[str, Any],
    llm: LLMParameters,
    config: MetadataDigestConfig | None = None,
) -> str:
    """Produce a trimmed representation of execution metadata for adapter prompts."""

    safe_metadata: Mapping[str, Any] = metadata or {}
    digest_cfg = config or MetadataDigestConfig()
    if not digest_cfg.enabled:
        return json.dumps(safe_metadata, separators=(",", ":"), ensure_ascii=False)

    budget = _resolve_budget(digest_cfg, llm.provider)
    builder = _PromptDigestBuilder(safe_metadata, digest_cfg, budget)
    digest_payload = builder.build()
    encoded = json.dumps(digest_payload, separators=(",", ":"), ensure_ascii=False)
    if len(encoded) > budget:
        raise PromptDigestTooLargeError(
            f"metadata digest exceeded provider budget ({len(encoded)} > {budget} chars)"
        )
    return encoded


def _resolve_budget(config: MetadataDigestConfig, provider: LLMProvider) -> int:
    if config.char_budget:
        return config.char_budget
    if provider in config.provider_char_budgets:
        return config.provider_char_budgets[provider]
    token_limit = PROVIDER_TOKEN_LIMITS.get(provider)
    if token_limit:
        calculated = int(token_limit * DEFAULT_CONTEXT_RATIO * AVERAGE_CHARS_PER_TOKEN)
        return max(MIN_CHAR_BUDGET, calculated)
    return DEFAULT_CHAR_BUDGET


def _sort_step_keys(keys: Iterable[str]) -> List[str]:
    def _key(value: str) -> Tuple[int, str]:
        if value.isdigit():
            return int(value), value
        try:
            return int(value.split("-", 1)[0]), value
        except (ValueError, IndexError):
            return math.inf, value

    return sorted(keys, key=_key)


class _PromptDigestBuilder:
    def __init__(self, metadata: Mapping[str, Any], config: MetadataDigestConfig, budget: int) -> None:
        self._metadata = metadata
        self._config = config
        self._budget = budget
        self._included_keys: set[str] = set()

    def build(self) -> Dict[str, Any]:
        digest: Dict[str, Any] = {}
        optional_sections: List[str] = []

        summary = self._build_summary_section()
        if summary:
            digest["summary"] = summary

        plan = self._build_plan_section()
        if plan:
            digest["plan"] = plan

        session = self._build_session_section()
        if session:
            digest["session"] = session

        learning = self._build_learning_section()
        if learning:
            digest["learning"] = learning
            optional_sections.append("learning")

        steps_section = self._build_recent_steps_section()
        if steps_section:
            digest["recent_steps"] = steps_section
            optional_sections.append("recent_steps")

        prompt_rewrite = self._build_prompt_rewrite_section()
        if prompt_rewrite:
            digest["prompt_rewrite"] = prompt_rewrite
            optional_sections.append("prompt_rewrite")

        playbooks = self._build_playbooks_section()
        if playbooks:
            digest["playbooks"] = playbooks
            optional_sections.append("playbooks")

        trajectory = self._build_trajectory_section()
        if trajectory:
            digest["trajectory"] = trajectory
            optional_sections.append("trajectory")

        digest_stats = {
            "budget": self._budget,
        }

        encoded = self._encode(digest)
        removed_sections: List[str] = []
        optional_order = ["prompt_rewrite", "playbooks", "trajectory", "learning", "recent_steps"]
        for name in optional_order:
            if len(encoded) <= self._budget:
                break
            if name in digest:
                removed_sections.append(name)
                digest.pop(name, None)
                encoded = self._encode(digest)
        if len(encoded) > self._budget:
            raise PromptDigestTooLargeError(
                f"metadata digest exceeded provider budget ({len(encoded)} > {self._budget} chars)"
            )

        section_sizes = {
            key: _encoded_length(value) for key, value in digest.items() if key != "digest_stats"
        }

        missing_keys = sorted(set(self._metadata.keys()) - self._included_keys)
        if missing_keys:
            digest_stats["omitted"] = missing_keys
        if removed_sections:
            digest_stats["omitted_sections"] = removed_sections
        digest_stats["sections"] = section_sizes
        digest["digest_stats"] = digest_stats

        encoded_with_stats = self._ensure_stats_within_budget(digest, digest_stats)
        final_size = len(encoded_with_stats)
        if final_size > self._budget:
            raise PromptDigestTooLargeError(
                f"metadata digest exceeded provider budget ({final_size} > {self._budget} chars)"
            )

        utilisation = final_size / self._budget if self._budget else 0.0
        if utilisation >= 0.75:
            logger.warning(
                "metadata digest consuming %.1f%% of provider budget (%s/%s chars)",
                utilisation * 100,
                final_size,
                self._budget,
            )
        else:
            logger.debug(
                "metadata digest size %s chars (%.1f%% of budget); section breakdown: %s",
                final_size,
                utilisation * 100,
                section_sizes,
            )
        return digest

    def _encode(self, payload: Mapping[str, Any]) -> str:
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

    def _ensure_stats_within_budget(self, digest: Dict[str, Any], digest_stats: Dict[str, Any]) -> str:
        encoded = self._finalise_stats(digest, digest_stats)
        if len(encoded) <= self._budget:
            return encoded

        for key in ("sections", "omitted", "omitted_sections"):
            if key in digest_stats:
                digest_stats.pop(key)
                encoded = self._finalise_stats(digest, digest_stats)
                if len(encoded) <= self._budget:
                    return encoded
        return encoded

    def _finalise_stats(self, digest: Dict[str, Any], digest_stats: Dict[str, Any]) -> str:
        previous_len = None
        encoded = self._encode(digest)
        for _ in range(4):
            current_len = len(encoded)
            if current_len == previous_len:
                break
            digest_stats["size"] = current_len
            digest_stats["util"] = round(current_len / self._budget, 6) if self._budget else 0.0
            previous_len = current_len
            encoded = self._encode(digest)
        return encoded

    def _build_summary_section(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {}
        task = self._metadata.get("task")
        if isinstance(task, str) and task.strip():
            summary["task"] = _truncate(task, self._config.max_section_chars)
            self._included_keys.add("task")
        execution_mode = self._metadata.get("execution_mode")
        if execution_mode:
            summary["execution_mode"] = execution_mode
            self._included_keys.add("execution_mode")
        active_actor = self._metadata.get("active_actor")
        if active_actor:
            summary["active_actor"] = active_actor
            self._included_keys.add("active_actor")
        single_shot = self._metadata.get("single_shot")
        if single_shot is not None:
            summary["single_shot"] = bool(single_shot)
            self._included_keys.add("single_shot")
        adaptive_summary = self._metadata.get("adaptive_summary")
        if adaptive_summary:
            summary["adaptive_summary"] = _truncate(str(adaptive_summary), self._config.max_section_chars)
            self._included_keys.add("adaptive_summary")

        adaptive = self._metadata.get("adaptive")
        if isinstance(adaptive, Mapping) and adaptive:
            summary["adaptive"] = _trim_mapping(adaptive, self._config.max_string_chars)
            self._included_keys.add("adaptive")

        triage = self._metadata.get("triage")
        if isinstance(triage, Mapping) and triage:
            summary["triage"] = _trim_mapping(triage, self._config.max_string_chars)
            self._included_keys.add("triage")

        return summary

    def _build_plan_section(self) -> Dict[str, Any] | None:
        plan = self._metadata.get("plan")
        if not isinstance(plan, Mapping):
            return None
        steps = plan.get("steps")
        summary: Dict[str, Any] = {"total_steps": len(steps) if isinstance(steps, Sequence) else 0}
        if isinstance(steps, Sequence):
            trimmed_steps: List[Dict[str, Any]] = []
            for step in steps[: self._config.max_plan_steps]:
                if not isinstance(step, Mapping):
                    continue
                entry = {
                    "id": step.get("id"),
                    "description": _truncate(str(step.get("description", "")), self._config.max_string_chars),
                }
                if step.get("tool"):
                    entry["tool"] = step["tool"]
                trimmed_steps.append(entry)
            if trimmed_steps:
                summary["steps"] = trimmed_steps
        self._included_keys.add("plan")
        return summary

    def _build_session_section(self) -> Dict[str, Any] | None:
        session_meta = self._metadata.get("session_metadata")
        session_section: Dict[str, Any] = {}
        if isinstance(session_meta, Mapping):
            allowed = set(self._config.include_session_keys)
            collected: Dict[str, Any] = {}
            for key, value in session_meta.items():
                if key in allowed:
                    collected[key] = _short_value(value, self._config.max_string_chars, self._config.max_section_chars)
                elif key.endswith("_stats"):
                    collected[key] = _short_value(value, self._config.max_string_chars, self._config.max_section_chars)
            if collected:
                session_section["session_metadata"] = collected
            self._included_keys.add("session_metadata")

        token_usage = self._metadata.get("token_usage")
        if isinstance(token_usage, Mapping):
            session_section["token_usage"] = dict(token_usage)
            self._included_keys.add("token_usage")

        reward = self._metadata.get("session_reward")
        if isinstance(reward, Mapping):
            session_section["session_reward"] = _summarise_reward(reward, self._config.max_string_chars)
            self._included_keys.add("session_reward")

        reward_stats = self._metadata.get("session_reward_stats")
        if isinstance(reward_stats, Mapping):
            session_section["session_reward_stats"] = dict(reward_stats)
            self._included_keys.add("session_reward_stats")

        reward_audit = self._metadata.get("session_reward_audit")
        if isinstance(reward_audit, Sequence) and not isinstance(reward_audit, (str, bytes)):
            session_section["session_reward_audit_summary"] = {
                "entries": len(reward_audit),
                "sample": _summarise_list(reward_audit, self._config.max_reward_audit_entries, self._config.max_string_chars),
            }
            self._included_keys.add("session_reward_audit")

        validation_cache = self._metadata.get("validation_cache")
        if isinstance(validation_cache, Mapping):
            session_section["validation_cache"] = {"keys": len(validation_cache)}
            self._included_keys.add("validation_cache")

        validation_blobs = self._metadata.get("validation_blobs")
        if isinstance(validation_blobs, Mapping):
            session_section["validation_blobs"] = {"keys": len(validation_blobs)}
            self._included_keys.add("validation_blobs")

        single_shot_results = self._metadata.get("single_shot_results")
        if isinstance(single_shot_results, Sequence) and not isinstance(single_shot_results, (str, bytes)):
            session_section["single_shot_results"] = {
                "count": len(single_shot_results),
                "sample": _summarise_list(
                    single_shot_results, min(2, len(single_shot_results)), self._config.max_string_chars
                ),
            }
            self._included_keys.add("single_shot_results")

        if not session_section:
            return None
        return session_section

    def _build_learning_section(self) -> Dict[str, Any] | None:
        learning: Dict[str, Any] = {}

        student = self._metadata.get("session_student_learning")
        if student:
            learning["student_learning"] = _truncate(str(student), self._config.max_section_chars)
            self._included_keys.add("session_student_learning")

        teacher = self._metadata.get("session_teacher_learning")
        if teacher:
            learning["teacher_learning"] = _truncate(str(teacher), self._config.max_section_chars)
            self._included_keys.add("session_teacher_learning")

        note = self._metadata.get("session_learning_note")
        if note:
            learning["session_note"] = _truncate(str(note), self._config.max_section_chars)
            self._included_keys.add("session_learning_note")

        learning_state = self._metadata.get("learning_state")
        if isinstance(learning_state, Mapping):
            learning["learning_state"] = _trim_mapping(learning_state, self._config.max_string_chars)
            self._included_keys.add("learning_state")

        learning_history = self._metadata.get("learning_history")
        if isinstance(learning_history, Mapping):
            entries = learning_history.get("entries")
            if isinstance(entries, Sequence):
                summary_entries = []
                for entry in entries[-self._config.max_learning_history_entries :]:
                    if not isinstance(entry, Mapping):
                        continue
                    summary_entries.append(_summarise_learning_entry(entry, self._config.max_string_chars))
                scores = [
                    entry.get("reward", {}).get("score")
                    for entry in entries
                    if isinstance(entry, Mapping) and isinstance(entry.get("reward"), Mapping)
                ]
                average_score = fmean([score for score in scores if isinstance(score, (int, float))]) if scores else None
                learning["learning_history"] = {
                    "entries": len(entries),
                    "recent": summary_entries,
                    "average_score": average_score,
                }
            self._included_keys.add("learning_history")

        learning_failures = self._metadata.get("learning_synthesis_failures")
        if isinstance(learning_failures, Sequence) and not isinstance(learning_failures, (str, bytes)):
            learning["learning_synthesis_failures"] = {
                "count": len(learning_failures),
                "last_failure": _truncate(str(learning_failures[-1]), self._config.max_section_chars)
                if learning_failures
                else None,
            }
            self._included_keys.add("learning_synthesis_failures")

        if not learning:
            return None
        return learning

    def _build_recent_steps_section(self) -> Sequence[Dict[str, Any]] | None:
        steps_meta = self._metadata.get("steps")
        if not isinstance(steps_meta, Mapping) or not steps_meta:
            return None
        ordered_keys = _sort_step_keys(list(steps_meta.keys()))
        summaries: List[Dict[str, Any]] = []
        for key in ordered_keys[-self._config.max_step_summaries :]:
            step_data = steps_meta.get(key)
            if not isinstance(step_data, Mapping):
                continue
            attempts = step_data.get("attempts")
            attempt_count = len(attempts) if isinstance(attempts, Sequence) else 0
            latest_attempt = attempts[-1] if attempt_count else {}
            evaluation = latest_attempt.get("evaluation") if isinstance(latest_attempt, Mapping) else {}
            validation = evaluation.get("validation") if isinstance(evaluation, Mapping) else {}
            reward = evaluation.get("reward") if isinstance(evaluation, Mapping) else {}
            summary = {
                "step_id": key,
                "attempts": attempt_count,
                "status": latest_attempt.get("status") if isinstance(latest_attempt, Mapping) else step_data.get("status"),
            }
            if isinstance(validation, Mapping) and "valid" in validation:
                summary["validation_passed"] = validation.get("valid")
            if isinstance(reward, Mapping) and "score" in reward:
                summary["reward_score"] = reward.get("score")
            if isinstance(reward, Mapping) and reward.get("rationale"):
                summary["reward_rationale"] = _truncate(str(reward.get("rationale")), self._config.max_string_chars)
            guidance = step_data.get("guidance")
            if isinstance(guidance, Sequence) and guidance:
                summary["last_guidance"] = _truncate(str(guidance[-1]), self._config.max_string_chars)
            summaries.append(summary)
        self._included_keys.add("steps")
        return summaries

    def _build_prompt_rewrite_section(self) -> Dict[str, Any] | None:
        prompt_rewrite = self._metadata.get("prompt_rewrite")
        if not isinstance(prompt_rewrite, Mapping):
            return None
        truncated = _trim_mapping(prompt_rewrite, self._config.max_prompt_rewrite_chars)
        self._included_keys.add("prompt_rewrite")
        learning_apply = self._metadata.get("learning_apply_to_prompts")
        payload: Dict[str, Any] = {"rewritten": truncated}
        if learning_apply is not None:
            payload["learning_applied"] = bool(learning_apply)
            self._included_keys.add("learning_apply_to_prompts")
        return payload

    def _build_playbooks_section(self) -> Dict[str, Any] | None:
        playbooks = self._metadata.get("_learning_playbooks")
        if not isinstance(playbooks, Mapping):
            return None
        payload = {
            "keys": list(playbooks.keys()),
            "count": len(playbooks),
        }
        self._included_keys.add("_learning_playbooks")
        return payload

    def _build_trajectory_section(self) -> Dict[str, Any] | None:
        trajectory = self._metadata.get("session_trajectory")
        if not isinstance(trajectory, Mapping):
            return None
        steps = trajectory.get("steps")
        payload: Dict[str, Any] = {
            "has_final_answer": bool(trajectory.get("final_answer")),
            "steps_recorded": len(steps) if isinstance(steps, Sequence) else 0,
            "teacher_intervened": trajectory.get("teacher_intervened"),
        }
        self._included_keys.add("session_trajectory")
        return payload


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    if limit <= 3:
        return value[:limit]
    return value[: limit - 3] + "..."


def _short_value(value: Any, string_limit: int, section_limit: int) -> Any:
    if isinstance(value, str):
        return _truncate(value, min(string_limit, section_limit))
    if isinstance(value, Mapping):
        return _trim_mapping(value, string_limit)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return _summarise_list(value, 3, string_limit)
    return value


def _trim_mapping(value: Mapping[str, Any], string_limit: int) -> Dict[str, Any]:
    trimmed: Dict[str, Any] = {}
    for key, val in list(value.items()):
        if isinstance(val, str):
            trimmed[key] = _truncate(val, string_limit)
        elif isinstance(val, Mapping):
            trimmed[key] = _trim_mapping(val, string_limit)
        elif isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
            trimmed[key] = _summarise_list(val, 3, string_limit)
        else:
            trimmed[key] = val
    return trimmed


def _summarise_list(values: Sequence[Any], limit: int, string_limit: int) -> List[Any]:
    if limit <= 0:
        return []
    collected: List[Any] = []
    for value in values[:limit]:
        if isinstance(value, str):
            collected.append(_truncate(value, string_limit))
        elif isinstance(value, Mapping):
            collected.append(_trim_mapping(value, string_limit))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            collected.append(_summarise_list(value, limit, string_limit))
        else:
            collected.append(value)
    if len(values) > limit:
        collected.append({"omitted": len(values) - limit})
    return collected


def _summarise_reward(payload: Mapping[str, Any], string_limit: int) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    if "score" in payload:
        summary["score"] = payload.get("score")
    if payload.get("rationale"):
        summary["rationale"] = _truncate(str(payload.get("rationale")), string_limit)
    judges = payload.get("judges")
    if isinstance(judges, Sequence):
        summary["judges"] = _summarise_list(judges, 2, string_limit)
    raw = payload.get("raw")
    if isinstance(raw, Mapping):
        summary["raw"] = _trim_mapping(raw, string_limit)
    return summary


def _summarise_learning_entry(entry: Mapping[str, Any], string_limit: int) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    reward = entry.get("reward")
    if isinstance(reward, Mapping):
        payload["reward"] = {
            "score": reward.get("score"),
            "rationale": _truncate(str(reward.get("rationale")), string_limit) if reward.get("rationale") else None,
        }
    for key in ("student_learning", "teacher_learning"):
        if entry.get(key):
            payload[key] = _truncate(str(entry.get(key)), string_limit)
    for key in ("created_at", "completed_at"):
        if entry.get(key):
            payload[key] = entry.get(key)
    return payload


def _encoded_length(value: Any) -> int:
    return len(json.dumps(value, separators=(",", ":"), ensure_ascii=False))
