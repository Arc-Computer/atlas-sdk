"""Runtime usage instrumentation for policy nuggets."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

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
        self._config = _TrackerConfig(
            enabled=bool(raw_config.get("enabled", True)),
            capture_examples=bool(raw_config.get("capture_examples", False)),
            max_examples=int(raw_config.get("max_examples_per_nugget", 2) or 0),
        )
        usage_store = metadata.setdefault("learning_usage", {})
        usage_store.setdefault("roles", {})
        session_block = usage_store.setdefault("session", {})
        session_block.setdefault("cue_hits", 0)
        session_block.setdefault("action_adoptions", 0)
        session_block.setdefault("unique_cue_steps", [])
        session_block.setdefault("unique_adoption_steps", [])
        usage_store.setdefault("detectors", {})
        self._usage_store = usage_store

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def register_nuggets(self, role: str, nuggets: Iterable[Dict[str, Any]]) -> None:
        if not self.enabled:
            return
        role_store = self._usage_store["roles"].setdefault(role, {})
        detector_store = self._usage_store["detectors"].setdefault(role, [])
        existing_detector_ids = {entry.get("nugget_id") for entry in detector_store}
        for nugget in nuggets or []:
            nugget_id = nugget.get("id")
            if not nugget_id:
                continue
            entry = role_store.setdefault(
                nugget_id,
                {
                    "cue_hits": 0,
                    "action_adoptions": 0,
                    "successful_adoptions": 0,
                    "step_ids": [],
                    "adoption_steps": [],
                    "runtime_handle": nugget.get("action", {}).get("runtime_handle"),
                    "category": nugget.get("scope", {}).get("category"),
                },
            )
            entry.setdefault("metadata", {})
            entry["metadata"].setdefault("cue", nugget.get("cue"))
            entry["metadata"].setdefault("scope", nugget.get("scope"))
            entry["metadata"].setdefault("expected_effect", nugget.get("expected_effect"))
            if nugget_id not in existing_detector_ids:
                detector_store.append(
                    {
                        "nugget_id": nugget_id,
                        "type": (nugget.get("cue", {}) or {}).get("type"),
                        "pattern": (nugget.get("cue", {}) or {}).get("pattern"),
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
            nugget_id = detector.get("nugget_id")
            pattern = detector.get("pattern")
            cue_type = str(detector.get("type") or "").lower()
            if not nugget_id or not pattern:
                continue
            if _cue_matches(cue_type, pattern, text):
                self.record_cue_hit(role, nugget_id, step_id=step_id, snippet=context_hint or text)
                matches.append({"nugget_id": nugget_id, "pattern": pattern, "type": cue_type})
        return matches

    def record_cue_hit(
        self,
        role: str,
        nugget_id: str,
        *,
        step_id: int | None,
        snippet: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        role_store = self._usage_store["roles"].setdefault(role, {})
        entry = role_store.get(nugget_id)
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
        for nugget_id, entry in role_store.items():
            if entry.get("runtime_handle") == runtime_handle:
                matched = True
                entry["action_adoptions"] = int(entry.get("action_adoptions", 0)) + 1
                if success:
                    entry["successful_adoptions"] = int(entry.get("successful_adoptions", 0)) + 1
                if step_id is not None and step_id not in entry["adoption_steps"]:
                    entry["adoption_steps"].append(step_id)
                if self._config.capture_examples and metadata:
                    examples = entry.setdefault("adoption_examples", [])
                    if len(examples) < self._config.max_examples:
                        examples.append(metadata)
        if matched:
            session_block = self._usage_store["session"]
            session_block["action_adoptions"] = int(session_block.get("action_adoptions", 0)) + 1
            if step_id is not None and step_id not in session_block["unique_adoption_steps"]:
                session_block["unique_adoption_steps"].append(step_id)

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


__all__ = ["LearningUsageTracker", "get_tracker"]
