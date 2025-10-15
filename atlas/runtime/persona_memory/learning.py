"""Candidate generation for persona memory continual learning."""

from __future__ import annotations

import json
import logging
import typing
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence
from uuid import UUID, uuid4

from atlas.runtime.orchestration.execution_context import ExecutionContext
from atlas.runtime.persona_memory.constants import canonical_persona_name
from atlas.runtime.persona_memory.fingerprint import FingerprintInputs
from atlas.types import Result, StepResult

if typing.TYPE_CHECKING:
    from atlas.runtime.storage.database import Database
logger = logging.getLogger(__name__)

MAX_INSTRUCTION_CHARS = 500
DEFAULT_REWARD_THRESHOLD = 0.6


@dataclass
class CandidateSpec:
    """Specification for a persona memory candidate."""

    persona: str
    instruction: Dict[str, Any]
    agent_name: str
    tenant_id: str
    trigger_fingerprint: str
    normalized_instruction: str
    reward_snapshot: Dict[str, Any] | None = None
    retry_count: int | None = None
    source_session_id: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


def _sanitize_text(value: str | None) -> str | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if len(text) > MAX_INSTRUCTION_CHARS:
        text = text[:MAX_INSTRUCTION_CHARS].rstrip()
    return text


def _instruction_from_guidance(guidance: str, extra: str | None = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"append": guidance}
    if extra:
        payload["context"] = extra
    return payload


def _normalize_instruction(payload: Any) -> str:
    if isinstance(payload, str):
        sanitized = _sanitize_text(payload) or payload.strip()
        payload = {"append": sanitized}
    elif not isinstance(payload, dict):
        payload = {"append": str(payload)}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _infer_persona(step_result: StepResult, step_metadata: dict[str, Any]) -> str:
    metadata = step_result.metadata or {}
    persona = metadata.get("persona_target") or metadata.get("persona") or metadata.get("actor")
    if isinstance(persona, str):
        return canonical_persona_name(persona)
    guidance_source = step_metadata.get("guidance_source")
    if isinstance(guidance_source, str):
        return canonical_persona_name(guidance_source)
    return canonical_persona_name("student")


def _extract_reward_snapshot(step_result: StepResult, step_metadata: dict[str, Any]) -> Dict[str, Any] | None:
    reward = getattr(step_result.evaluation, "reward", None)
    if reward is not None and hasattr(reward, "to_dict"):
        snapshot = reward.to_dict()
    else:
        snapshot = None
    attempts_meta = step_metadata.get("attempts") or []
    if attempts_meta:
        latest = attempts_meta[-1]
        maybe_reward = latest.get("evaluation", {}).get("reward") if isinstance(latest, dict) else None
        if isinstance(maybe_reward, dict):
            snapshot = maybe_reward
    return snapshot


def extract_candidates(context: ExecutionContext, result: Result) -> List[CandidateSpec]:
    """Analyse execution context and result to derive persona memory candidates."""
    fingerprint_inputs = context.metadata.get("persona_fingerprint_inputs")
    fingerprint_hash = context.metadata.get("persona_fingerprint")
    if not isinstance(fingerprint_inputs, FingerprintInputs) or not fingerprint_hash:
        return []
    agent_name = fingerprint_inputs.agent_name
    tenant_id = fingerprint_inputs.tenant_id
    steps_metadata: dict[int, dict[str, Any]] = context.metadata.get("steps", {}) or {}
    threshold = context.metadata.get("persona_reward_threshold", DEFAULT_REWARD_THRESHOLD)
    seen: set[tuple[str, str, str]] = set()
    candidates: list[CandidateSpec] = []
    session_student_learning = context.metadata.get("session_student_learning")

    for step in result.step_results:
        step_meta = steps_metadata.get(step.step_id, {})
        attempts_meta = step_meta.get("attempts") or []
        guidance_list: Sequence[str] = step_meta.get("guidance") or []

        # Use step-level guidance if available, otherwise use session-level RIM learning
        guidance_text = _sanitize_text(
            guidance_list[-1] if guidance_list
            else step.metadata.get("guidance") if step.metadata
            else session_student_learning
        )

        if not guidance_text:
            continue

        retry_count = step.attempts or len(attempts_meta) or None
        reward = getattr(step.evaluation, "reward", None)
        score = getattr(reward, "score", None)
        should_generate = False
        if isinstance(score, (int, float)) and score < threshold:
            should_generate = True
        if retry_count and retry_count > 1:
            should_generate = True
        if guidance_list:
            should_generate = True
        if not should_generate:
            continue
        extra_context = None
        reason = None
        for attempt in reversed(attempts_meta):
            if isinstance(attempt, dict):
                reason = attempt.get("reason")
                if reason:
                    break
        if not reason and step.metadata:
            reason = step.metadata.get("reason")
        if not reason and attempts_meta:
            latest_attempt = attempts_meta[-1]
            if isinstance(latest_attempt, dict):
                reason = latest_attempt.get("status")
        extra_context = _sanitize_text(reason)
        instruction = _instruction_from_guidance(guidance_text, extra_context)
        normalized = _normalize_instruction(instruction)
        persona = _infer_persona(step, step_meta)
        dedup_key = (persona, fingerprint_hash, normalized)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        tags = _compose_candidate_tags(context, persona)
        candidate_metadata = {
            "tags": tags,
            "helpful_count": 0,
            "harmful_count": 0,
            "neutral_count": 0,
            "last_mode": context.metadata.get("adaptive", {}).get("active_mode"),
            "last_reward": None,
            "last_reward_at": None,
        }
        candidates.append(
            CandidateSpec(
                persona=persona,
                instruction=instruction,
                agent_name=agent_name,
                tenant_id=tenant_id,
                trigger_fingerprint=fingerprint_hash,
                normalized_instruction=normalized,
                reward_snapshot=_extract_reward_snapshot(step, step_meta),
                retry_count=retry_count,
                tags=tags,
                metadata=candidate_metadata,
            )
        )
    return candidates


async def write_candidates(database: "Database", session_id: int, candidates: Iterable[CandidateSpec]) -> List[UUID]:
    """Persist persona memory candidates, avoiding duplicates."""
    created_ids: list[UUID] = []
    grouped: dict[tuple[str, str, str, str], list[CandidateSpec]] = defaultdict(list)
    for candidate in candidates:
        canonical_persona = canonical_persona_name(candidate.persona)
        candidate.persona = canonical_persona
        key = (candidate.agent_name, candidate.tenant_id, canonical_persona, candidate.trigger_fingerprint)
        grouped[key].append(candidate)
    for key, specs in grouped.items():
        agent_name, tenant_id, persona, fingerprint = key
        existing = await database.fetch_persona_memories(agent_name, tenant_id, persona, fingerprint, statuses=["candidate"])
        existing_normalized = {
            _normalize_instruction(record.get("instruction"))
            for record in existing
            if record.get("instruction") is not None
        }
        for spec in specs:
            if spec.normalized_instruction in existing_normalized:
                continue
            memory_id = uuid4()
            existing_normalized.add(spec.normalized_instruction)
            spec.source_session_id = session_id
            record = {
                "memory_id": memory_id,
                "agent_name": spec.agent_name,
                "tenant_id": spec.tenant_id,
                "persona": spec.persona,
                "trigger_fingerprint": spec.trigger_fingerprint,
                "instruction": spec.instruction,
                "source_session_id": spec.source_session_id,
                "reward_snapshot": spec.reward_snapshot,
                "retry_count": spec.retry_count,
                "metadata": {**spec.metadata, "tags": spec.tags},
                "status": "candidate",
            }
            await database.create_persona_memory(record)
            created_ids.append(memory_id)
    return created_ids


def _compose_candidate_tags(context: ExecutionContext, persona: str) -> List[str]:
    tags: set[str] = {f"persona:{persona}"}
    adaptive_meta = context.metadata.get("adaptive", {}) if isinstance(context.metadata, dict) else {}
    default_tags = context.metadata.get("adaptive_default_tags", [])
    if isinstance(default_tags, (list, tuple)):
        for tag in default_tags:
            if isinstance(tag, str) and tag.strip():
                tags.add(tag.strip())
    dossier = context.metadata.get("triage", {}).get("dossier") if isinstance(context.metadata, dict) else None
    if isinstance(dossier, dict):
        for tag in dossier.get("tags", []) or []:
            if isinstance(tag, str) and tag.strip():
                tags.add(tag.strip())
    mode = adaptive_meta.get("active_mode")
    if isinstance(mode, str) and mode:
        tags.add(f"mode:{mode}")
    tags.add(f"persona:{persona}")
    return sorted(tags)
