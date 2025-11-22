"""Runtime usage instrumentation for playbook entries.

This module provides learning tracking for Atlas agents, enabling measurement
of which playbook entries are referenced during execution and which actions
are adopted based on learning recommendations.

Core Concepts:
    - Cue Detection: Identifying when user input matches learning cue patterns
    - Action Adoption: Tracking when tools are executed based on playbook entries
    - Impact Metrics: Computing effectiveness of learning entries across sessions

Typical Usage (BYOA Adapters):
    ```python
    from atlas.learning.playbook import resolve_playbook
    from atlas.learning.usage import get_tracker

    # 1. Retrieve playbook (auto-registers entries)
    playbook, digest, metadata = resolve_playbook("student", apply=True)

    # 2. Get tracker
    tracker = get_tracker()

    # 3. Detect cue hits
    tracker.detect_and_record("student", user_input, step_id=1)

    # 4. Record action adoptions
    tracker.record_action_adoption("student", "tool_name", success=True, step_id=1)

    # 5. Record final outcome
    tracker.record_session_outcome(reward_score=0.85)
    ```

See docs/sdk/learning_tracking.md for complete guide.
"""

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
    """Lightweight helper that records cue hits and action adoption statistics.

    This tracker instruments learning usage during agent execution, enabling
    measurement of:
    - Which playbook entries have their cues detected in user input
    - Which tools/actions are adopted based on playbook recommendations
    - Whether adopted actions succeed or fail
    - Session-level metrics (reward, tokens, retries, failures)

    The tracker is automatically created and attached to ExecutionContext.
    Access it via `get_tracker()` or `get_tracker(context)`.

    Usage Pattern:
        1. Register entries: Called automatically by resolve_playbook()
        2. Detect cues: Call detect_and_record() on user input
        3. Track adoptions: Call record_action_adoption() after tool execution
        4. Record outcome: Call record_session_outcome() at session end

    Attributes:
        enabled (bool): Whether tracking is active (from config)

    Methods:
        register_entries(): Register playbook entries for tracking
        detect_and_record(): Detect and record cue hits from text
        record_cue_hit(): Record a specific cue hit manually
        record_action_adoption(): Record when a tool/action is adopted
        record_session_outcome(): Record final session metrics
        snapshot(): Export current tracking state
    """

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
        """Register playbook entries for tracking.

        This method is called automatically by resolve_playbook() so BYOA adapters
        typically don't need to call it manually.

        For each entry, this method:
        - Creates tracking storage (cue_hits, action_adoptions, step_ids)
        - Registers cue detection patterns
        - Stores runtime_handle for action adoption matching

        Args:
            role: "student" or "teacher"
            entries: List of playbook entry dicts, each containing:
                - id: Unique entry identifier
                - cue: Dict with type and pattern for detection
                - action: Dict with runtime_handle for adoption tracking
                - scope: Category and other metadata

        Note:
            Called automatically by resolve_playbook() in atlas/learning/playbook.py
        """
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
        """Detect and record cue hits from text.

        Scans the provided text for patterns matching registered playbook entry cues.
        For each match, increments cue hit counters and optionally saves examples.

        Cue Types:
            - keyword: Simple substring match (case-insensitive)
            - regex: Regular expression pattern match
            - predicate: Condition-based match (treated as keyword)

        When to Call:
            - On user input/task description (planning phase)
            - On step descriptions (execution phase)
            - On any text that might contain learning cues

        Args:
            role: "student" or "teacher"
            text: Text to scan for cue patterns
            step_id: Optional step identifier for tracking which steps have cue hits
            context_hint: Optional text snippet to save as example (truncated to 200 chars)

        Returns:
            List of matched cue dicts, each containing:
                - entry_id: Which playbook entry was matched
                - pattern: The pattern that matched
                - type: Cue type (keyword, regex, predicate)

        Example:
            >>> tracker = get_tracker()
            >>> matches = tracker.detect_and_record("student", "create a new contact", step_id=1)
            >>> print(f"Detected {len(matches)} cue hits")
        """
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
        """Record when a tool/action is adopted based on playbook recommendation.

        This method tracks whether learning entries actually change agent behavior.
        It matches the runtime_handle parameter against playbook entries and
        increments adoption counters when matches are found.

        CRITICAL: The runtime_handle must EXACTLY match the action.runtime_handle
        stored in playbook entries. This is how Atlas links tool execution to
        learning recommendations.

        When to Call:
            - After executing any tool that might be in playbook entries
            - After taking any action recommended by learning
            - Typically called once per tool execution

        Args:
            role: "student" or "teacher"
            runtime_handle: Tool name or action identifier (must match playbook entries)
            success: Whether the action succeeded (True) or failed (False)
            step_id: Optional step identifier for tracking adoption per step
            metadata: Optional dict with additional context (saved as example if
                     capture_examples is enabled)

        Example:
            >>> tracker = get_tracker()
            >>> # After executing a tool
            >>> tracker.record_action_adoption(
            ...     "student",
            ...     runtime_handle="create_contact",  # Must match entry's runtime_handle
            ...     success=True,
            ...     step_id=1,
            ...     metadata={"tool_name": "create_contact", "args": {...}}
            ... )

        Troubleshooting:
            If adoptions remain 0 despite tool execution:
            1. Verify runtime_handle exactly matches playbook entry's action.runtime_handle
            2. Ensure playbook entries are registered (call resolve_playbook first)
            3. Check that learning synthesis has generated entries
        """
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
        """Record final session metrics and outcome.

        Call this method once at the end of session execution to record overall
        metrics used for computing learning entry impact.

        When to Call:
            - At the end of session execution, before returning results
            - After all steps have been executed
            - Typically called once per session

        Args:
            reward_score: Session reward score (0.0-1.0)
            token_usage: Dict with token counts:
                - total_tokens: Total tokens used
                - prompt_tokens: Input tokens
                - completion_tokens: Output tokens
                - calls: Number of LLM calls
            incident_id: Optional incident/task identifier for grouping
            task_identifier: Optional task type/category identifier
            incident_tags: Optional list of tags for categorization
            retry_count: Number of retries during session
            failure_flag: Whether session failed overall
            failure_events: List of failure event dicts

        Example:
            >>> tracker = get_tracker()
            >>> tracker.record_session_outcome(
            ...     reward_score=0.85,
            ...     token_usage={
            ...         "total_tokens": 2000,
            ...         "prompt_tokens": 1500,
            ...         "completion_tokens": 500,
            ...         "calls": 3
            ...     },
            ...     task_identifier="security-review",
            ...     retry_count=1,
            ...     failure_flag=False
            ... )
        """
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
        """Return the current usage store (already JSON-serializable).

        This method exports all tracked learning usage data for the current session,
        including:
        - Per-entry statistics (cue hits, adoptions, success/failure counts)
        - Session-level aggregates (total cues, total adoptions, reward score)
        - Step-level tracking (which steps had cue hits or adoptions)

        The returned dict is automatically persisted to the database and used for
        generating learning impact reports.

        Returns:
            Dict with structure:
                {
                    "roles": {
                        "student": {
                            "entry_id_1": {
                                "cue_hits": int,
                                "action_adoptions": int,
                                "successful_adoptions": int,
                                "failed_adoptions": int,
                                "step_ids": List[int],
                                ...
                            },
                            ...
                        }
                    },
                    "session": {
                        "cue_hits": int,
                        "action_adoptions": int,
                        "reward_score": float,
                        "token_usage": {...},
                        ...
                    }
                }
        """

        return dict(self._usage_store)


def get_tracker(context: ExecutionContext | None = None) -> LearningUsageTracker:
    """Get the learning usage tracker for the current execution context.

    This helper creates or retrieves the tracker instance attached to the
    execution context. Call this in BYOA adapters to access learning tracking.

    Args:
        context: Optional ExecutionContext. If None, uses ExecutionContext.get()

    Returns:
        LearningUsageTracker instance for the current session

    Raises:
        Exception: If ExecutionContext is not available (e.g., standalone testing)

    Example:
        >>> from atlas.learning.usage import get_tracker
        >>> from atlas.runtime.orchestration.execution_context import ExecutionContext
        >>>
        >>> context = ExecutionContext.get()
        >>> tracker = get_tracker(context)
        >>> # Or simply:
        >>> tracker = get_tracker()

    Note:
        Each execution context has its own tracker instance. Tracking data is
        stored in context.metadata["learning_usage"] and persisted to database
        at session end.
    """

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
