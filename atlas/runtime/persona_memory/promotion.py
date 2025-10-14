"""Promotion, cap enforcement, and compaction for persona memories."""

from __future__ import annotations

import json
import logging
import typing
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set
from uuid import UUID

from atlas.config.models import AtlasConfig
from atlas.runtime.persona_memory.constants import CANONICAL_PERSONAS, canonical_persona_name, persona_aliases
from atlas.runtime.persona_memory.fingerprint import FingerprintInputs
from atlas.runtime.persona_memory.learning import _sanitize_text  # type: ignore[attr-defined]

if typing.TYPE_CHECKING:
    from atlas.runtime.storage.database import Database

logger = logging.getLogger(__name__)

DEFAULT_PROMOTION_MIN_SAMPLES = 2
DEFAULT_PROMOTION_MIN_DELTA = 0.05
DEFAULT_PERSONA_CAPS = {
    "student": 5,
    "teacher_plan_review": 2,
    "teacher_validation": 4,
}
DEFAULT_FALLBACK_CAP = 3


def _normalize_instruction(payload: Any) -> str:
    """Normalise instruction payloads to a comparable JSON string."""
    if isinstance(payload, str):
        payload = {"append": _sanitize_text(payload) or payload.strip()}
    elif not isinstance(payload, dict):
        payload = {"append": str(payload)}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _parse_json(payload: Any) -> Any:
    if payload is None:
        return None
    if isinstance(payload, (dict, list)):
        return payload
    try:
        return json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError):
        return payload


def _score_from_snapshot(snapshot: Any) -> Optional[float]:
    snapshot = _parse_json(snapshot)
    if not isinstance(snapshot, dict):
        return None
    if isinstance(snapshot.get("score"), (int, float)):
        return float(snapshot["score"])
    reward = snapshot.get("reward")
    if isinstance(reward, dict) and isinstance(reward.get("score"), (int, float)):
        return float(reward["score"])
    return None


def _score_from_usage(entries: Sequence[Mapping[str, Any]]) -> Optional[float]:
    weighted: List[tuple[float, float]] = []
    for entry in entries:
        reward = _parse_json(entry.get("reward"))
        score = None
        if isinstance(reward, dict):
            if isinstance(reward.get("score"), (int, float)):
                score = float(reward["score"])
            elif isinstance(reward.get("reward"), dict) and isinstance(reward["reward"].get("score"), (int, float)):
                score = float(reward["reward"]["score"])
        if score is not None:
            weight = _mode_weight(entry.get("mode"))
            weighted.append((score, weight))
    if not weighted:
        return None
    numerator = sum(score * weight for score, weight in weighted)
    denominator = sum(weight for _, weight in weighted)
    if denominator <= 0:
        denominator = float(len(weighted))
    return numerator / denominator


def _retry_from_usage(entries: Sequence[Mapping[str, Any]]) -> Optional[float]:
    retries: List[float] = []
    for entry in entries:
        retry_value = entry.get("retry_count")
        if isinstance(retry_value, (int, float)):
            retries.append(float(retry_value))
    if not retries:
        return None
    return sum(retries) / len(retries)


def _mode_weight(mode: Any) -> float:
    if not isinstance(mode, str):
        return 1.0
    normalised = mode.strip().lower()
    if normalised == "auto":
        return 1.2
    if normalised == "paired":
        return 1.0
    if normalised == "coach":
        return 0.85
    if normalised == "escalate":
        return 0.7
    return 1.0


@dataclass
class PromotionSettings:
    min_samples: int = DEFAULT_PROMOTION_MIN_SAMPLES
    min_delta: float = DEFAULT_PROMOTION_MIN_DELTA
    caps: Dict[str, int] = field(default_factory=lambda: DEFAULT_PERSONA_CAPS.copy())
    compaction_enabled: bool = True

    def cap_for(self, persona: str) -> int:
        return self.caps.get(persona, self.caps.get("*", DEFAULT_FALLBACK_CAP))


@dataclass
class PromotionResult:
    promoted: List[str] = field(default_factory=list)
    demoted: List[str] = field(default_factory=list)
    compacted: List[str] = field(default_factory=list)
    invalidate_personas: Set[str] = field(default_factory=set)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "promoted": list(self.promoted),
            "demoted": list(self.demoted),
            "compacted": list(self.compacted),
            "invalidate_personas": list(self.invalidate_personas),
        }


def get_promotion_settings(config: AtlasConfig | None) -> PromotionSettings:
    settings = PromotionSettings()
    if config is None:
        return settings
    persona_cfg = {}
    if getattr(config, "metadata", None):
        candidate = config.metadata.get("persona_memory")
        if isinstance(candidate, dict):
            persona_cfg = candidate
    if persona_cfg:
        min_samples = persona_cfg.get("promotion_samples")
        if isinstance(min_samples, int) and min_samples > 0:
            settings.min_samples = min_samples
        min_delta = persona_cfg.get("promotion_threshold")
        if isinstance(min_delta, (int, float)):
            settings.min_delta = float(min_delta)
        caps_cfg = persona_cfg.get("persona_caps") or persona_cfg.get("caps")
        if isinstance(caps_cfg, dict):
            merged = settings.caps.copy()
            for key, value in caps_cfg.items():
                canonical_key = canonical_persona_name(key)
                try:
                    merged[canonical_key] = int(value)
                except (TypeError, ValueError):
                    continue
            settings.caps = merged
        compaction_flag = persona_cfg.get("compaction_enabled")
        if isinstance(compaction_flag, str):
            settings.compaction_enabled = compaction_flag.strip().lower() not in {"0", "false", "no"}
        elif compaction_flag is not None:
            settings.compaction_enabled = bool(compaction_flag)
    return settings


async def promote_and_compact(
    database: "Database",
    fingerprint_inputs: FingerprintInputs,
    fingerprint_hash: str,
    settings: PromotionSettings,
) -> PromotionResult:
    result = PromotionResult()
    if not fingerprint_hash:
        return result
    agent_name = fingerprint_inputs.agent_name
    tenant_id = fingerprint_inputs.tenant_id
    persona_order = list(CANONICAL_PERSONAS)

    for persona in persona_order:
        score_overrides: Dict[UUID, Optional[float]] = {}
        # Promote candidates for this persona
        candidates: List[Dict[str, Any]] = []
        for alias in persona_aliases(persona):
            records = await database.fetch_persona_memories(
                agent_name, tenant_id, alias, fingerprint_hash, statuses=["candidate"]
            )
            if records:
                candidates.extend(records)
        candidate_ids: List[UUID] = [record["memory_id"] for record in candidates]
        usage_map = await database.fetch_persona_usage(candidate_ids)
        promoted_ids: List[UUID] = []

        for record in candidates:
            memory_id: UUID = record["memory_id"]
            usage_entries = usage_map.get(memory_id, [])
            metadata = record.get("metadata") if isinstance(record, dict) else {}
            helpful_count = int((metadata or {}).get("helpful_count", 0) or 0)
            harmful_count = int((metadata or {}).get("harmful_count", 0) or 0)
            neutral_count = int((metadata or {}).get("neutral_count", 0) or 0)
            total_feedback = helpful_count + harmful_count + neutral_count
            if len(usage_entries) < settings.min_samples and total_feedback < settings.min_samples:
                continue
            baseline_score = _score_from_snapshot(record.get("reward_snapshot"))
            usage_score = _score_from_usage(usage_entries)
            baseline_retry = record.get("retry_count")
            usage_retry = _retry_from_usage(usage_entries)

            improved = False
            if helpful_count >= settings.min_samples and helpful_count > harmful_count:
                improved = True
            if usage_score is not None and baseline_score is not None:
                improved = usage_score >= baseline_score + settings.min_delta
            if not improved and isinstance(baseline_retry, (int, float)) and usage_retry is not None:
                improved = usage_retry < float(baseline_retry)
            if not improved and usage_score is not None and baseline_score is None:
                improved = True  # No baseline, but we have positive evidence

            if improved:
                await database.update_persona_status([memory_id], "active")
                promoted_ids.append(memory_id)
                score_overrides[memory_id] = usage_score
                result.promoted.append(str(memory_id))
                result.invalidate_personas.add(persona)

        # Enforce caps & compaction on active memories
        active_records: List[Mapping[str, Any]] = []
        for alias in persona_aliases(persona):
            records = await database.fetch_persona_memories(
                agent_name, tenant_id, alias, fingerprint_hash, statuses=["active"]
            )
            if records:
                active_records.extend(records)
        if promoted_ids:
            promoted_usage = await database.fetch_persona_usage(promoted_ids)
            for promoted_id, entries in promoted_usage.items():
                score_overrides[promoted_id] = _score_from_usage(entries) or score_overrides.get(promoted_id)

        cap = settings.cap_for(persona)
        if cap > 0 and len(active_records) > cap:
            sorted_records = _sort_active_records(active_records, score_overrides)
            overflow = sorted_records[cap:]
            overflow_ids = [record["memory_id"] for record in overflow]
            await database.update_persona_status(overflow_ids, "replaced")
            result.demoted.extend(str(memory_id) for memory_id in overflow_ids)
            if overflow_ids:
                result.invalidate_personas.add(persona)
            # keep only top cap records for further compaction logic
            active_records = sorted_records[:cap]

        if not settings.compaction_enabled or not active_records:
            continue

        normalized_map: MutableMapping[str, List[Mapping[str, Any]]] = {}
        for record in active_records:
            instruction = _parse_json(record.get("instruction"))
            normalized = _normalize_instruction(instruction)
            normalized_map.setdefault(normalized, []).append(record)

        for records in normalized_map.values():
            if len(records) <= 1:
                continue
            # Retain the most recently updated instruction
            sorted_group = sorted(
                records,
                key=lambda item: (item.get("updated_at"), _score_from_snapshot(item.get("reward_snapshot")) or 0.0),
                reverse=True,
            )
            keeper = sorted_group[0]
            demote_rest = [entry for entry in sorted_group[1:]]
            demote_ids = [entry["memory_id"] for entry in demote_rest]
            if demote_ids:
                await database.update_persona_status(demote_ids, "replaced")
                result.compacted.extend(str(memory_id) for memory_id in demote_ids)
                result.invalidate_personas.add(persona)

    if result.promoted or result.demoted or result.compacted:
        logger.info(
            "Persona promotion summary | promoted=%s demoted=%s compacted=%s personas=%s",
            result.promoted,
            result.demoted,
            result.compacted,
            list(result.invalidate_personas),
        )
    return result


def _sort_active_records(
    records: Sequence[Mapping[str, Any]],
    score_overrides: Mapping[UUID, Optional[float]] | None = None,
) -> List[Mapping[str, Any]]:
    overrides = score_overrides or {}

    def sort_key(record: Mapping[str, Any]):
        memory_id = record.get("memory_id")
        override = overrides.get(memory_id) if memory_id is not None else None
        meta = record.get("metadata") if isinstance(record, Mapping) else None
        metadata_score = None
        if isinstance(meta, dict):
            metadata_score = meta.get("last_reward")
        score = override if override is not None else metadata_score
        if score is None:
            score = _score_from_snapshot(record.get("reward_snapshot")) or 0.0
        helpful = int(meta.get("helpful_count", 0) if isinstance(meta, dict) else 0)
        harmful = int(meta.get("harmful_count", 0) if isinstance(meta, dict) else 0)
        confidence = helpful - harmful
        updated = record.get("updated_at")
        return (score, confidence, updated)

    return sorted(records, key=sort_key, reverse=True)
