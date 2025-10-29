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
    PolicyGateRules,
    PolicyRubricWeights,
    PolicySchemaConfig,
)
from atlas.learning.nuggets import evaluate_candidates, normalise_candidates, stabilise_nugget_id
from atlas.learning.prompts import LEARNING_SYNTHESIS_PROMPT
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
    policy_nuggets: List[Dict[str, Any]] | None = None
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
        self._schema: PolicySchemaConfig = config.schema
        self._rubric_weights: PolicyRubricWeights = config.rubric_weights
        self._gate_rules: PolicyGateRules = config.gates
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
        try:
            response = await client.acomplete(
                messages,
                response_format={"type": "json_object"},
            )
            audit_entry = {
                "model": client.model,
                "messages": messages,
                "response": response.content,
                "reasoning": response.reasoning or {},
                "raw_response": response.raw,
            }
        except Exception as exc:
            logger.warning("Learning synthesis call failed for %s: %s", learning_key, exc)
            return None

        parsed = self._try_parse_json(response.content)
        if parsed is None:
            logger.warning("Learning synthesis returned non-JSON payload for %s", learning_key)
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
            current_nuggets = metadata_payload.get("policy_nuggets")
            if isinstance(current_nuggets, list):
                payload["current_policy_nuggets"] = current_nuggets
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

        policy_eval = self._evaluate_policy_candidates(payload, current_metadata, context)
        metadata = policy_eval["metadata"]
        accepted = policy_eval["accepted"]

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
            policy_nuggets=metadata.get("policy_nuggets") if isinstance(metadata, dict) else None,
            rubric_summary=policy_eval.get("summary"),
            gate_failures=policy_eval.get("failures") or None,
        )
        return result

    def _evaluate_policy_candidates(
        self,
        payload: Dict[str, Any],
        current_metadata: Dict[str, Any],
        context: ExecutionContext,
    ) -> Dict[str, Any]:
        metadata = copy.deepcopy(current_metadata) if isinstance(current_metadata, dict) else {}
        incoming_metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None
        if isinstance(incoming_metadata, dict):
            metadata.update(copy.deepcopy(incoming_metadata))
        baseline_nuggets = []
        if isinstance(metadata.get("policy_nuggets"), list):
            baseline_nuggets = copy.deepcopy(metadata.get("policy_nuggets"))
        baseline_by_id = {
            entry.get("id"): entry for entry in baseline_nuggets if isinstance(entry, dict) and entry.get("id")
        }

        raw_candidates = payload.get("policy_nuggets")
        if not isinstance(raw_candidates, list):
            raw_candidates = []
        handles = self._resolve_runtime_handles(context)

        candidates = normalise_candidates(raw_candidates, schema=self._schema, default_audience="student")
        for candidate in candidates:
            candidate_id = candidate.get("id")
            if candidate_id and candidate_id in baseline_by_id:
                baseline_audience = baseline_by_id[candidate_id].get("audience")
                if baseline_audience:
                    candidate["audience"] = baseline_audience

        evaluations, summary = evaluate_candidates(
            candidates,
            gates=self._gate_rules,
            weights=self._rubric_weights,
            schema=self._schema,
            allowed_handles=handles["exact"],
            allowed_prefixes=handles["prefixes"],
        )
        summary["weights"] = self._normalised_weights_map()
        summary["accepted"] = True

        failures: List[Dict[str, Any]] = []
        for item in evaluations:
            if not item.passed():
                summary["accepted"] = False
                failures.append(
                    {
                        "id": item.nugget.get("id"),
                        "gates": item.evaluation.gates,
                        "scores": item.evaluation.scores,
                        "reasons": item.evaluation.failure_reasons,
                    }
                )

        accepted = summary["accepted"]
        timestamp = datetime.now(timezone.utc).isoformat()
        metadata["policy_version"] = self._schema.version
        metadata["last_evaluation"] = {
            "timestamp": timestamp,
            "summary": summary,
            "candidates": [
                {
                    "id": item.nugget.get("id"),
                    "audience": item.nugget.get("audience"),
                    "rubric": item.evaluation.to_dict(),
                }
                for item in evaluations
            ],
        }

        if accepted and not evaluations and not raw_candidates:
            metadata.setdefault("policy_nuggets", baseline_nuggets)
            metadata["policy_summary"] = summary
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
            for item in evaluations:
                if not item.passed():
                    continue
                nugget_payload = item.to_metadata()
                nugget_payload.pop("sequence", None)
                nugget_id = nugget_payload.get("id") or stabilise_nugget_id(nugget_payload)
                nugget_payload["id"] = nugget_id
                if not nugget_payload.get("audience"):
                    nugget_payload["audience"] = "student"
                scope = nugget_payload.get("scope")
                if not isinstance(scope, dict):
                    scope = {}
                    nugget_payload["scope"] = scope
                if not scope.get("category"):
                    prior_scope = baseline_by_id.get(nugget_id, {}).get("scope") if nugget_id in baseline_by_id else {}
                    category = (prior_scope or {}).get("category") or self._schema.default_scope_category
                    scope["category"] = category
                nugget_payload["rubric"]["weights"] = self._normalised_weights_map()
                prior_entry = baseline_by_id.get(nugget_id)
                nugget_payload["provenance"] = self._build_provenance(
                    nugget_payload,
                    prior_entry,
                    context,
                    lifecycle="active",
                )
                active_entries.append(nugget_payload)
                seen_ids.add(nugget_id)
            for nugget_id, prior in baseline_by_id.items():
                if nugget_id in seen_ids:
                    continue
                stale = copy.deepcopy(prior)
                stale["provenance"] = self._build_provenance(stale, prior, context, lifecycle="deprecated")
                active_entries.append(stale)
            metadata["policy_nuggets"] = active_entries
            metadata["policy_summary"] = summary
            metadata["last_updated_at"] = timestamp
        else:
            metadata["policy_summary"] = summary
            metadata.setdefault("policy_nuggets", baseline_nuggets)
            metadata["last_failure"] = {
                "timestamp": timestamp,
                "failures": failures,
                "rejected_candidates": [
                    {
                        "id": item.nugget.get("id") or stabilise_nugget_id(item.nugget),
                        "audience": item.nugget.get("audience"),
                        "scope": item.nugget.get("scope"),
                        "status": {
                            "category": (item.nugget.get("scope") or {}).get("category") or self._schema.default_scope_category,
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

    def _resolve_runtime_handles(self, context: ExecutionContext) -> Dict[str, List[str]]:
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
        return {"exact": unique_handles, "prefixes": prefixes}

    def _normalised_weights_map(self) -> Dict[str, float]:
        raw = {
            "actionability": max(self._rubric_weights.actionability, 0.0),
            "generality": max(self._rubric_weights.generality, 0.0),
            "hookability": max(self._rubric_weights.hookability, 0.0),
            "concision": max(self._rubric_weights.concision, 0.0),
        }
        total = sum(raw.values()) or 1.0
        return {key: round(value / total, 4) for key, value in raw.items()}

    def _build_provenance(
        self,
        nugget_payload: Dict[str, Any],
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
        scope = nugget_payload.get("scope") if isinstance(nugget_payload.get("scope"), dict) else {}
        category = (scope or {}).get("category")
        if not isinstance(category, str) or not category.strip():
            if isinstance(prior_entry, dict):
                prior_scope = prior_entry.get("scope") if isinstance(prior_entry.get("scope"), dict) else {}
                category = prior_scope.get("category")
            category = category or self._schema.default_scope_category
            if isinstance(scope, dict):
                scope["category"] = category
                nugget_payload["scope"] = scope
        provenance["status"] = {
            "category": category,
            "lifecycle": lifecycle,
        }
        provenance["rubric"] = copy.deepcopy(nugget_payload.get("rubric"))
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

    @staticmethod
    def _clean_str(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip()
        return str(value).strip()


__all__ = ["LearningSynthesizer", "LearningSynthesisResult"]
