"""Learning synthesizer that maintains persistent pamphlets across sessions."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List

from atlas.config.models import (
    LearningConfig,
    LLMParameters,
    PlaybookEntryGateRules,
    PlaybookEntryRubricWeights,
    PlaybookEntrySchemaConfig,
)
from atlas.learning.playbook_entries import (
    evaluate_playbook_entries,
    normalise_playbook_entries,
    stabilise_playbook_entry_id,
)
from atlas.learning.prompts import LEARNING_SYNTHESIS_PROMPT
from atlas.learning.schema import build_playbook_entry_schema
from atlas.runtime.orchestration.execution_context import ExecutionContext
from atlas.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class LearningSynthesisResult:
    """Structured output returned by the learning synthesizer."""

    student_learning: str | None
    teacher_learning: str | None
    learning_state: Dict[str, Any]
    session_note: str | None = None
    audit: Dict[str, Any] | None = None
    playbook_entries: List[Dict[str, Any]] | None = None
    rubric_summary: Dict[str, Any] | None = None
    gate_failures: List[Dict[str, Any]] | None = None


class LearningSynthesizer:
    """Generates updated learning pamphlets using an LLM."""

    def __init__(
        self,
        config: LearningConfig,
        *,
        client: LLMClient | None = None,
        fallback_llm: LLMParameters | None = None,
    ) -> None:
        self._config = config
        self._prompt = (config.prompts.synthesizer if config.prompts and config.prompts.synthesizer else LEARNING_SYNTHESIS_PROMPT)
        self._schema: PlaybookEntrySchemaConfig = config.schema
        self._rubric_weights: PlaybookEntryRubricWeights = config.rubric_weights
        self._gate_rules: PlaybookEntryGateRules = config.gates
        llm_params = config.llm or fallback_llm
        if config.enabled and llm_params is None and client is None:
            raise ValueError("learning.llm must be configured when the learning synthesizer is enabled")
        self._client = client or (LLMClient(llm_params) if llm_params is not None else None)

    @property
    def enabled(self) -> bool:
        return bool(self._config.enabled and self._client is not None)

    async def asynthesize(
        self,
        *,
        learning_key: str,
        task: str,
        reward: Dict[str, Any] | None,
        trajectory: Dict[str, Any] | None,
        learning_state: Dict[str, Any] | None,
        history: Dict[str, Any] | None,
    ) -> LearningSynthesisResult | None:
        if not self.enabled:
            logger.debug("Learning synthesizer disabled; skipping update for %s", learning_key)
            return None
        if not self._config.update_enabled:
            logger.debug("Learning updates disabled via configuration; skipping update for %s", learning_key)
            return None

        context = ExecutionContext.get()
        context.metadata["active_actor"] = "learning"
        context.metadata["_reasoning_origin"] = ("learning", "synthesis")

        payload = self._build_payload(task, reward, trajectory, learning_state, history)

        # Inject available runtime handles for LLM to use
        handles = self._resolve_runtime_handles(context)
        payload["available_runtime_handles"] = handles["exact"]

        messages = [
            {"role": "system", "content": self._prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        response = None
        audit_entry: Dict[str, Any] | None = None
        client = self._client
        if client is None:
            logger.debug("Learning synthesizer client unavailable; skipping update for %s", learning_key)
            return None
        
        # Build JSON schema for structured outputs (Gemini models)
        json_schema = build_playbook_entry_schema()
        overrides: Dict[str, Any] = {}
        if self._is_gemini_model(client.model):
            # Pass JSON schema via extra_body for Gemini structured outputs
            overrides["extra_body"] = {
                "response_json_schema": json_schema
            }
        
        try:
            response = await client.acomplete(
                messages,
                response_format={"type": "json_object"},
                overrides=overrides,
            )
            audit_entry = {
                "model": client.model,
                "messages": messages,
                "response": response.content,
                "reasoning": response.reasoning or {},
                "raw_response": response.raw,
                "structured_output": self._is_gemini_model(client.model),
            }
        except Exception as exc:
            logger.warning("Learning synthesis call failed for %s: %s", learning_key, exc)
            return None

        parsed = self._try_parse_json(response.content)
        if parsed is None:
            logger.error(
                "Learning synthesis returned non-JSON payload for %s (model: %s). "
                "Response preview: %s",
                learning_key,
                client.model,
                response.content[:200] if response.content else "empty",
            )
            return None

        result = self._build_result(parsed, learning_state or {})
        if audit_entry is not None:
            result.audit = audit_entry
            context.metadata.setdefault("session_learning_audit", []).append(audit_entry)
        reasoning_queue = context.metadata.get("_llm_reasoning_queue", [])
        if reasoning_queue:
            context.metadata["_llm_reasoning_queue"] = []
            if audit_entry is not None:
                audit_entry["reasoning_queue"] = list(reasoning_queue)
        return result

    def synthesize(
        self,
        *,
        learning_key: str,
        task: str,
        reward: Dict[str, Any] | None,
        trajectory: Dict[str, Any] | None,
        learning_state: Dict[str, Any] | None,
        history: Dict[str, Any] | None,
    ) -> LearningSynthesisResult | None:
        if not self.enabled:
            return None
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.asynthesize(
                    learning_key=learning_key,
                    task=task,
                    reward=reward,
                    trajectory=trajectory,
                    learning_state=learning_state,
                    history=history,
                )
            )
        raise RuntimeError("LearningSynthesizer.synthesize cannot be invoked inside an active event loop")

    def _build_payload(
        self,
        task: str,
        reward: Dict[str, Any] | None,
        trajectory: Dict[str, Any] | None,
        learning_state: Dict[str, Any] | None,
        history: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        latest_session: Dict[str, Any] = {
            "task": task,
            "reward": reward or {},
            "evidence": trajectory or {},
        }
        state_payload = learning_state or {}
        metadata_payload = state_payload.get("metadata") if isinstance(state_payload, dict) else None
        pamphlets = {
            "student_pamphlet": state_payload.get("student_learning") if isinstance(state_payload, dict) else None,
            "teacher_pamphlet": state_payload.get("teacher_learning") if isinstance(state_payload, dict) else None,
        }
        payload: Dict[str, Any] = {
            "pamphlets": pamphlets,
            "latest_session": latest_session,
        }
        if isinstance(metadata_payload, dict):
            payload["pamphlet_metadata"] = metadata_payload
            current_entries = metadata_payload.get("playbook_entries")
            if isinstance(current_entries, list):
                payload["current_playbook_entries"] = current_entries
        if history:
            payload["history"] = self._trim_history(history)
        return payload

    def _trim_history(self, history: Dict[str, Any]) -> Dict[str, Any]:
        limit = self._config.history_limit
        if not isinstance(history, dict):
            return {}
        entries = history.get("entries")
        if isinstance(entries, list) and limit and limit > 0:
            history = dict(history)
            history["entries"] = entries[-limit:]
        return history

    @staticmethod
    def _try_parse_json(payload: Any) -> Dict[str, Any] | None:
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, str):
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                return None
        return None

    def _build_result(self, payload: Dict[str, Any], baseline_state: Dict[str, Any]) -> LearningSynthesisResult:
        context = ExecutionContext.get()
        session_student = self._clean_str(payload.get("session_student_learning"))
        session_teacher = self._clean_str(payload.get("session_teacher_learning"))
        candidate_student = self._clean_str(payload.get("student_pamphlet"))
        candidate_teacher = self._clean_str(payload.get("teacher_pamphlet"))

        current_student = baseline_state.get("student_learning") if isinstance(baseline_state, dict) else ""
        current_teacher = baseline_state.get("teacher_learning") if isinstance(baseline_state, dict) else None
        current_metadata = baseline_state.get("metadata") if isinstance(baseline_state, dict) else {}
        if not isinstance(current_metadata, dict):
            current_metadata = {}

        # Prune ineffective entries from existing playbook before evaluating new candidates
        existing_entries = current_metadata.get("playbook_entries")
        if isinstance(existing_entries, list) and existing_entries:
            pruned_entries = self._prune_ineffective_entries(
                existing_entries,
                min_sessions=self._config.pruning_config.min_sessions,
            )
            current_metadata["playbook_entries"] = pruned_entries

        playbook_eval = self._evaluate_playbook_entries(payload, current_metadata, context)
        metadata = playbook_eval["metadata"]
        accepted = playbook_eval["accepted"]

        student_pamphlet = candidate_student if accepted and candidate_student is not None else (current_student or "")
        teacher_pamphlet = candidate_teacher if accepted and candidate_teacher is not None else current_teacher

        session_note = None
        if session_student or session_teacher:
            parts: List[str] = []
            if session_student:
                parts.append(f"Student: {session_student}")
            if session_teacher:
                parts.append(f"Teacher: {session_teacher}")
            session_note = " ".join(parts)

        learning_state = {
            "student_learning": student_pamphlet,
            "teacher_learning": teacher_pamphlet,
            "metadata": metadata,
        }
        result = LearningSynthesisResult(
            student_learning=session_student,
            teacher_learning=session_teacher,
            learning_state=learning_state,
            session_note=session_note,
            playbook_entries=metadata.get("playbook_entries") if isinstance(metadata, dict) else None,
            rubric_summary=playbook_eval.get("summary"),
            gate_failures=playbook_eval.get("failures") or None,
        )
        return result

    def _evaluate_playbook_entries(
        self,
        payload: Dict[str, Any],
        current_metadata: Dict[str, Any],
        context: ExecutionContext,
    ) -> Dict[str, Any]:
        metadata = copy.deepcopy(current_metadata) if isinstance(current_metadata, dict) else {}
        incoming_metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None
        if isinstance(incoming_metadata, dict):
            metadata.update(copy.deepcopy(incoming_metadata))
        baseline_entries: List[Dict[str, Any]] = []
        if isinstance(metadata.get("playbook_entries"), list):
            baseline_entries = copy.deepcopy(metadata.get("playbook_entries"))
        baseline_entries_by_id = {
            entry.get("id"): entry for entry in baseline_entries if isinstance(entry, dict) and entry.get("id")
        }

        raw_candidates = payload.get("playbook_entries")
        if not isinstance(raw_candidates, list):
            raw_candidates = []
        handles = self._resolve_runtime_handles(context)

        candidates = normalise_playbook_entries(raw_candidates, schema=self._schema, default_audience="student")
        for candidate in candidates:
            candidate_id = candidate.get("id")
            if candidate_id and candidate_id in baseline_entries_by_id:
                baseline_audience = baseline_entries_by_id[candidate_id].get("audience")
                if baseline_audience:
                    candidate["audience"] = baseline_audience

        evaluations, summary = evaluate_playbook_entries(
            candidates,
            gates=self._gate_rules,
            weights=self._rubric_weights,
            schema=self._schema,
            allowed_handles=handles["exact"],
            allowed_prefixes=handles["prefixes"],
            allow_missing_mapping=handles.get("allow_missing_mapping", False),
        )
        summary["weights"] = self._normalised_weights_map()
        
        # Separate blocking failures (semantic correctness) from non-blocking warnings (generality)
        # Actionability and cue gates are blocking - entries must pass these
        # Generality gate is non-blocking - entries can be accepted provisionally
        blocking_failures: List[Dict[str, Any]] = []
        provisional_warnings: List[Dict[str, Any]] = []
        
        for item in evaluations:
            if not item.passed():
                gates = item.evaluation.gates
                # Actionability and cue are blocking (semantic correctness)
                if not gates.get("actionability", True) or not gates.get("cue", True):
                    blocking_failures.append(
                        {
                            "id": item.entry.get("id"),
                            "gates": gates,
                            "scores": item.evaluation.scores,
                            "reasons": item.evaluation.failure_reasons,
                        }
                    )
                # Generality is non-blocking (empirical validation will decide)
                elif not gates.get("generality", True):
                    provisional_warnings.append(
                        {
                            "id": item.entry.get("id"),
                            "gates": gates,
                            "scores": item.evaluation.scores,
                            "reasons": item.evaluation.failure_reasons,
                        }
                    )
        
        # Accept if no blocking failures (allow provisional entries)
        accepted = len(blocking_failures) == 0
        summary["accepted"] = accepted
        summary["provisional_count"] = len(provisional_warnings)
        summary["blocking_failures"] = len(blocking_failures)
        failures = blocking_failures + provisional_warnings
        timestamp = datetime.now(timezone.utc).isoformat()
        metadata["playbook_version"] = self._schema.version
        metadata["last_evaluation"] = {
            "timestamp": timestamp,
            "summary": summary,
            "candidates": [
                {
                    "id": item.entry.get("id"),
                    "audience": item.entry.get("audience"),
                    "rubric": item.evaluation.to_dict(),
                }
                for item in evaluations
            ],
        }

        if accepted and not evaluations and not raw_candidates:
            metadata.setdefault("playbook_entries", baseline_entries)
            metadata["playbook_summary"] = summary
            metadata.setdefault("last_updated_at", timestamp)
            return {
                "metadata": metadata,
                "accepted": True,
                "summary": summary,
                "failures": failures,
            }

        if accepted:
            active_entries: List[Dict[str, Any]] = []
            seen_ids: set[str] = set()
            # Create mapping of entry IDs to their warnings for provisional entries
            provisional_by_id = {w.get("id"): w for w in provisional_warnings}
            
            for item in evaluations:
                # Skip only blocking failures - accept entries with provisional warnings
                gates = item.evaluation.gates
                if not gates.get("actionability", True) or not gates.get("cue", True):
                    continue
                    
                entry_payload = item.to_metadata()
                entry_payload.pop("sequence", None)
                entry_id = entry_payload.get("id") or stabilise_playbook_entry_id(entry_payload)
                entry_payload["id"] = entry_id
                if not entry_payload.get("audience"):
                    entry_payload["audience"] = "student"
                scope = entry_payload.get("scope")
                if not isinstance(scope, dict):
                    scope = {}
                    entry_payload["scope"] = scope
                if not scope.get("category"):
                    prior_scope = baseline_entries_by_id.get(entry_id, {}).get("scope") if entry_id in baseline_entries_by_id else {}
                    category = (prior_scope or {}).get("category") or self._schema.default_scope_category
                    scope["category"] = category
                entry_payload["rubric"]["weights"] = self._normalised_weights_map()
                
                # Mark provisional entries with validation status and warnings
                if entry_id in provisional_by_id:
                    warning = provisional_by_id[entry_id]
                    entry_metadata = entry_payload.setdefault("metadata", {})
                    entry_metadata["validation_status"] = "provisional"
                    entry_metadata["validation_warnings"] = warning.get("reasons", [])
                    lifecycle = "provisional"
                else:
                    entry_metadata = entry_payload.setdefault("metadata", {})
                    entry_metadata["validation_status"] = "validated"
                    lifecycle = "active"
                
                prior_entry = baseline_entries_by_id.get(entry_id)
                entry_payload["provenance"] = self._build_provenance(
                    entry_payload,
                    prior_entry,
                    context,
                    lifecycle=lifecycle,
                )
                if isinstance(prior_entry, dict):
                    prior_impact = prior_entry.get("impact")
                    if isinstance(prior_impact, dict):
                        entry_payload["impact"] = copy.deepcopy(prior_impact)
                active_entries.append(entry_payload)
                seen_ids.add(entry_id)
            for entry_id, prior in baseline_entries_by_id.items():
                if entry_id in seen_ids:
                    continue
                # Skip pruned entries
                prior_provenance = prior.get("provenance", {})
                prior_status = prior_provenance.get("status", {})
                if prior_status.get("lifecycle") == "pruned":
                    continue
                stale = copy.deepcopy(prior)
                stale["provenance"] = self._build_provenance(stale, prior, context, lifecycle="deprecated")
                active_entries.append(stale)
            metadata["playbook_entries"] = active_entries
            metadata["playbook_summary"] = summary
            metadata["last_updated_at"] = timestamp
            
            # Filter out pruned entries from final active list
            metadata["playbook_entries"] = [
                entry for entry in active_entries
                if entry.get("provenance", {}).get("status", {}).get("lifecycle") != "pruned"
            ]
        else:
            metadata["playbook_summary"] = summary
            metadata.setdefault("playbook_entries", baseline_entries)
            metadata["last_failure"] = {
                "timestamp": timestamp,
                "failures": failures,
                "rejected_candidates": [
                    {
                        "id": item.entry.get("id") or stabilise_playbook_entry_id(item.entry),
                        "audience": item.entry.get("audience"),
                        "scope": item.entry.get("scope"),
                        "status": {
                            "category": (item.entry.get("scope") or {}).get("category") or self._schema.default_scope_category,
                            "lifecycle": "rejected",
                        },
                        "rubric": item.evaluation.to_dict(),
                    }
                    for item in evaluations
                    if not item.passed()
                ],
            }
            context.metadata.setdefault("learning_synthesis_failures", []).append(
                {
                    "timestamp": timestamp,
                    "failures": failures,
                    "summary": summary,
                }
            )

        return {
            "metadata": metadata,
            "accepted": accepted,
            "summary": summary,
            "failures": failures,
        }

    def _resolve_runtime_handles(self, context: ExecutionContext) -> Dict[str, Any]:
        metadata = context.metadata if isinstance(context.metadata, dict) else {}
        handles: List[str] = []
        configured = self._schema.allowed_runtime_handles or []
        handles.extend(configured)
        runtime_handles = metadata.get("runtime_handles") or metadata.get("available_runtime_handles") or []
        if isinstance(runtime_handles, list):
            handles.extend(str(handle) for handle in runtime_handles if isinstance(handle, str))
        prefixes = list(self._schema.runtime_handle_prefixes or [])
        unique_handles: List[str] = []
        seen: set[str] = set()
        for handle in handles:
            lowered = handle.lower()
            if lowered not in seen:
                seen.add(lowered)
                unique_handles.append(handle)

        # Auto-enable missing tool mapping when no tools available (tool-less agents)
        has_tools = bool(unique_handles or prefixes)

        return {
            "exact": unique_handles,
            "prefixes": prefixes,
            "allow_missing_mapping": not has_tools,
        }

    def _normalised_weights_map(self) -> Dict[str, float]:
        raw = {
            "actionability": max(self._rubric_weights.actionability, 0.0),
            "generality": max(self._rubric_weights.generality, 0.0),
            "hookability": max(self._rubric_weights.hookability, 0.0),
            "concision": max(self._rubric_weights.concision, 0.0),
        }
        total = sum(raw.values()) or 1.0
        return {key: round(value / total, 4) for key, value in raw.items()}

    def _prune_ineffective_entries(
        self,
        entries: List[Dict[str, Any]],
        min_sessions: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Prune entries based on empirical impact metrics.
        
        Pruning criteria:
        1. Too specific: cue_hit_rate < min_cue_hit_rate AND sessions_observed >= min_sessions
        2. Harmful: reward_delta < -min_reward_delta AND sessions_with_hits >= 5
        3. Neutral: reward_delta < min_reward_delta AND adoption_rate > 0.5 AND sessions_with_hits >= 10
        4. No transfer: transfer_success = False AND sessions_observed >= min_transfer_sessions
        """
        pruning_config = self._config.pruning_config
        min_cue_hit_rate = pruning_config.min_cue_hit_rate
        min_reward_delta = pruning_config.min_reward_delta
        min_transfer_sessions = pruning_config.min_transfer_sessions
        
        pruned = []
        for entry in entries:
            if not isinstance(entry, dict):
                pruned.append(entry)
                continue
                
            impact = entry.get("impact", {})
            if not isinstance(impact, dict):
                pruned.append(entry)  # Keep - no impact data yet
                continue
                
            sessions_observed = impact.get("sessions_observed", 0)
            
            # Not enough data - keep entry
            if sessions_observed < min_sessions:
                pruned.append(entry)
                continue
                
            # Compute metrics
            sessions_with_hits = impact.get("sessions_with_hits", 0)
            cue_hit_rate = sessions_with_hits / sessions_observed if sessions_observed > 0 else 0.0
            
            total_cue_hits = impact.get("total_cue_hits", 0)
            successful_adoptions = impact.get("successful_adoptions", 0)
            adoption_rate = successful_adoptions / total_cue_hits if total_cue_hits > 0 else 0.0
            
            reward_with_sum = impact.get("reward_with_sum", 0.0)
            reward_with_count = impact.get("reward_with_count", 0)
            reward_without_sum = impact.get("reward_without_sum", 0.0)
            reward_without_count = impact.get("reward_without_count", 0)
            
            reward_with = reward_with_sum / reward_with_count if reward_with_count > 0 else None
            reward_without = reward_without_sum / reward_without_count if reward_without_count > 0 else None
            reward_delta = (reward_with - reward_without) if (reward_with is not None and reward_without is not None) else None
            
            incident_ids = impact.get("incident_ids", [])
            transfer_success = len(incident_ids) >= 2 if isinstance(incident_ids, list) else False
            
            # Pruning rules
            prune_reason = None
            
            # 1. Too specific - rarely fires
            if cue_hit_rate < min_cue_hit_rate and sessions_observed >= min_sessions:
                prune_reason = "too_specific"
            
            # 2. Harmful - negative reward delta
            elif reward_delta is not None and reward_delta < -min_reward_delta and sessions_with_hits >= 5:
                prune_reason = "harmful"
            
            # 3. Neutral - no improvement despite adoption
            elif (
                reward_delta is not None
                and reward_delta < min_reward_delta
                and adoption_rate > 0.5
                and sessions_with_hits >= 10
            ):
                prune_reason = "neutral"
            
            # 4. No transfer - only fires in one context
            elif not transfer_success and sessions_observed >= min_transfer_sessions:
                prune_reason = "no_transfer"
            
            if prune_reason:
                # Mark as pruned but keep in list for audit trail
                provenance = entry.get("provenance")
                if not isinstance(provenance, dict):
                    provenance = {}
                    entry["provenance"] = provenance
                status = provenance.get("status")
                if not isinstance(status, dict):
                    status = {}
                    provenance["status"] = status
                status["lifecycle"] = "pruned"
                provenance["prune_reason"] = prune_reason
                provenance["pruned_at"] = datetime.now(timezone.utc).isoformat()
                # Don't add to active entries - they'll be filtered out
                continue
            
            pruned.append(entry)
        
        return pruned

    def _build_provenance(
        self,
        entry_payload: Dict[str, Any],
        prior_entry: Dict[str, Any] | None,
        context: ExecutionContext,
        *,
        lifecycle: str,
    ) -> Dict[str, Any]:
        timestamp = datetime.now(timezone.utc).isoformat()
        provenance: Dict[str, Any] = {}
        if isinstance(prior_entry, dict):
            prior_provenance = prior_entry.get("provenance")
            if isinstance(prior_provenance, dict):
                provenance = copy.deepcopy(prior_provenance)
        if "created_at" not in provenance:
            provenance["created_at"] = timestamp
        provenance["updated_at"] = timestamp
        session_id = context.metadata.get("session_id") if isinstance(context.metadata, dict) else None
        if session_id is not None:
            provenance["source_session_id"] = provenance.get("source_session_id", session_id)
        guidance_digest = self._teacher_guidance_digest(context)
        if guidance_digest:
            provenance["teacher_guidance_digest"] = guidance_digest
            provenance["source_teacher_intervention_hash"] = guidance_digest
        scope = entry_payload.get("scope") if isinstance(entry_payload.get("scope"), dict) else {}
        category = (scope or {}).get("category")
        if not isinstance(category, str) or not category.strip():
            if isinstance(prior_entry, dict):
                prior_scope = prior_entry.get("scope") if isinstance(prior_entry.get("scope"), dict) else {}
                category = prior_scope.get("category")
            category = category or self._schema.default_scope_category
            if isinstance(scope, dict):
                scope["category"] = category
                entry_payload["scope"] = scope
        provenance["status"] = {
            "category": category,
            "lifecycle": lifecycle,
        }
        provenance["rubric"] = copy.deepcopy(entry_payload.get("rubric"))
        provenance["weights"] = self._normalised_weights_map()
        return provenance

    def _teacher_guidance_digest(self, context: ExecutionContext) -> str | None:
        metadata = context.metadata if isinstance(context.metadata, dict) else {}
        steps = metadata.get("steps")
        if not isinstance(steps, dict):
            return None
        notes: List[str] = []
        for entry in steps.values():
            if not isinstance(entry, dict):
                continue
            guidance = entry.get("guidance")
            if isinstance(guidance, list):
                notes.extend(str(item).strip() for item in guidance if isinstance(item, str) and item.strip())
        if not notes:
            return None
        serialized = "\n".join(sorted(set(notes)))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    def _is_gemini_model(self, model: str) -> bool:
        """Check if model is a Gemini model.
        
        Args:
            model: Model identifier string
            
        Returns:
            True if model is a Gemini model, False otherwise
        """
        return model.startswith("gemini/") or model.startswith("google/")

    @staticmethod
    def _clean_str(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip()
        return str(value).strip()


__all__ = ["LearningSynthesizer", "LearningSynthesisResult"]
