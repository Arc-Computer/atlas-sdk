"""Runtime usage instrumentation for playbook entries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from atlas.runtime.orchestration.execution_context import ExecutionContext


@dataclass(slots=True)
class _TrackerConfig:
    enabled: bool
    capture_examples: bool
    max_examples: int


class LearningUsageTracker:
    """Lightweight helper that records cue hits and action adoption statistics."""

    def __init__(self, context: ExecutionContext) -> None:
        self._context = context
        metadata = context.metadata
        raw_config = metadata.get("learning_usage_config") or {}
        max_examples_config = raw_config.get("max_examples_per_entry", 2)
        self._config = _TrackerConfig(
            enabled=bool(raw_config.get("enabled", True)),
            capture_examples=bool(raw_config.get("capture_examples", False)),
            max_examples=int(max_examples_config or 0),
        )
        usage_store = metadata.setdefault("learning_usage", {})
        usage_store.setdefault("roles", {})
        session_block = usage_store.setdefault("session", {})
        session_block.setdefault("cue_hits", 0)
        session_block.setdefault("action_adoptions", 0)
        session_block.setdefault("failed_adoptions", 0)
        session_block.setdefault("unique_cue_steps", [])
        session_block.setdefault("unique_adoption_steps", [])
        session_block.setdefault("reward_score", None)
        session_block.setdefault("token_usage", {})
        session_block.setdefault("incident_id", None)
        session_block.setdefault("task_identifier", None)
        session_block.setdefault("incident_tags", [])
        session_block.setdefault("retry_count", 0)
        session_block.setdefault("failure_flag", False)
        session_block.setdefault("failure_events", [])
        usage_store.setdefault("detectors", {})
        self._usage_store = usage_store

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def register_entries(self, role: str, entries: Iterable[Dict[str, Any]]) -> None:
        if not self.enabled:
            return
        role_store = self._usage_store["roles"].setdefault(role, {})
        detector_store = self._usage_store["detectors"].setdefault(role, [])
        existing_detector_ids = {entry.get("entry_id") for entry in detector_store}
        for entry_payload in entries or []:
            entry_id = entry_payload.get("id")
            if not entry_id:
                continue
            entry_store = role_store.setdefault(
                entry_id,
                {
                    "cue_hits": 0,
                    "action_adoptions": 0,
                    "successful_adoptions": 0,
                    "failed_adoptions": 0,
                    "step_ids": [],
                    "adoption_steps": [],
                    "runtime_handle": entry_payload.get("action", {}).get("runtime_handle"),
                    "category": entry_payload.get("scope", {}).get("category"),
                },
            )
            entry_store.setdefault("metadata", {})
            entry_store["metadata"].setdefault("cue", entry_payload.get("cue"))
            entry_store["metadata"].setdefault("scope", entry_payload.get("scope"))
            entry_store["metadata"].setdefault("expected_effect", entry_payload.get("expected_effect"))
            if entry_id not in existing_detector_ids:
                detector_store.append(
                    {
                        "entry_id": entry_id,
                        "type": (entry_payload.get("cue", {}) or {}).get("type"),
                        "pattern": (entry_payload.get("cue", {}) or {}).get("pattern"),
                    }
                )

    def detect_and_record(
        self,
        role: str,
        text: str,
        *,
        step_id: int | None = None,
        context_hint: str | None = None,
    ) -> List[Dict[str, Any]]:
        if not self.enabled or not text:
            return []
        detectors = self._usage_store.get("detectors", {}).get(role) or []
        matches: List[Dict[str, Any]] = []
        for detector in detectors:
            entry_id = detector.get("entry_id")
            pattern = detector.get("pattern")
            cue_type = str(detector.get("type") or "").lower()
            if not entry_id or not pattern:
                continue
            if _cue_matches(cue_type, pattern, text):
                self.record_cue_hit(role, entry_id, step_id=step_id, snippet=context_hint or text)
                matches.append({"entry_id": entry_id, "pattern": pattern, "type": cue_type})
        return matches

    def record_cue_hit(
        self,
        role: str,
        entry_id: str,
        *,
        step_id: int | None,
        snippet: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        role_store = self._usage_store["roles"].setdefault(role, {})
        entry = role_store.get(entry_id)
        if entry is None:
            return
        entry["cue_hits"] = int(entry.get("cue_hits", 0)) + 1
        session_block = self._usage_store["session"]
        session_block["cue_hits"] = int(session_block.get("cue_hits", 0)) + 1
        if step_id is not None:
            if step_id not in entry["step_ids"]:
                entry["step_ids"].append(step_id)
            if step_id not in session_block["unique_cue_steps"]:
                session_block["unique_cue_steps"].append(step_id)
        if self._config.capture_examples and snippet:
            examples = entry.setdefault("cue_examples", [])
            if len(examples) < self._config.max_examples:
                examples.append(snippet.strip()[:200])

    def record_action_adoption(
        self,
        role: str,
        runtime_handle: str | None,
        *,
        success: bool,
        step_id: int | None = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.enabled:
            return
        if not runtime_handle:
            return
        role_store = self._usage_store["roles"].setdefault(role, {})
        matched = False
        for entry_id, entry in role_store.items():
            if entry.get("runtime_handle") == runtime_handle:
                matched = True
                entry["action_adoptions"] = int(entry.get("action_adoptions", 0)) + 1
                if success:
                    entry["successful_adoptions"] = int(entry.get("successful_adoptions", 0)) + 1
                else:
                    entry["failed_adoptions"] = int(entry.get("failed_adoptions", 0)) + 1
                if step_id is not None and step_id not in entry["adoption_steps"]:
                    entry["adoption_steps"].append(step_id)
                if self._config.capture_examples and metadata:
                    examples = entry.setdefault("adoption_examples", [])
                    if len(examples) < self._config.max_examples:
                        examples.append(metadata)
        if matched:
            session_block = self._usage_store["session"]
            session_block["action_adoptions"] = int(session_block.get("action_adoptions", 0)) + 1
            if not success:
                session_block["failed_adoptions"] = int(session_block.get("failed_adoptions", 0)) + 1
            if step_id is not None and step_id not in session_block["unique_adoption_steps"]:
                session_block["unique_adoption_steps"].append(step_id)

    def record_session_outcome(
        self,
        *,
        reward_score: float | None = None,
        token_usage: Optional[Dict[str, Any]] = None,
        incident_id: str | None = None,
        task_identifier: str | None = None,
        incident_tags: Sequence[str] | None = None,
        retry_count: int | None = None,
        failure_flag: bool | None = None,
        failure_events: Sequence[Dict[str, Any]] | None = None,
    ) -> None:
        if not self.enabled:
            return
        session_block = self._usage_store["session"]
        if reward_score is not None:
            session_block["reward_score"] = float(reward_score)
        if isinstance(token_usage, dict):
            tracked = session_block.setdefault("token_usage", {})
            prompt_tokens = token_usage.get("prompt_tokens")
            completion_tokens = token_usage.get("completion_tokens")
            total_tokens = token_usage.get("total_tokens")
            calls = token_usage.get("calls")
            if prompt_tokens is not None:
                numeric = _as_number(prompt_tokens)
                if numeric is not None:
                    tracked["prompt_tokens"] = numeric
            if completion_tokens is not None:
                numeric = _as_number(completion_tokens)
                if numeric is not None:
                    tracked["completion_tokens"] = numeric
            if total_tokens is not None:
                numeric = _as_number(total_tokens)
                if numeric is not None:
                    tracked["total_tokens"] = numeric
            if calls is not None:
                numeric = _as_number(calls)
                if numeric is not None:
                    tracked["calls"] = numeric
        if incident_id is not None:
            session_block["incident_id"] = incident_id
        if task_identifier is not None:
            session_block["task_identifier"] = task_identifier
        if incident_tags:
            existing = session_block.setdefault("incident_tags", [])
            for tag in incident_tags:
                if not isinstance(tag, str):
                    continue
                normalised = tag.strip()
                if normalised and normalised not in existing:
                    existing.append(normalised)
        if retry_count is not None:
            session_block["retry_count"] = int(retry_count)
        if failure_flag is not None:
            session_block["failure_flag"] = bool(failure_flag)
        if failure_events:
            cleaned: list[Dict[str, Any]] = []
            for event in failure_events:
                if isinstance(event, dict):
                    cleaned.append({key: value for key, value in event.items() if key and value is not None})
            if cleaned:
                session_block["failure_events"] = cleaned

    def snapshot(self) -> Dict[str, Any]:
        """Return the current usage store (already JSON-serialisable)."""

        return dict(self._usage_store)


def get_tracker(context: ExecutionContext | None = None) -> LearningUsageTracker:
    """Helper to fetch the tracker for the active execution context."""

    context = context or ExecutionContext.get()
    return LearningUsageTracker(context)


def _cue_matches(cue_type: str, pattern: str, text: str) -> bool:
    if cue_type == "regex":
        try:
            return bool(re.search(pattern, text, flags=re.IGNORECASE))
        except re.error:
            return False
    lowered = text.lower()
    if cue_type in {"keyword", "predicate"}:
        return pattern.lower() in lowered
    return False


def _as_number(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = ["LearningUsageTracker", "get_tracker"]
