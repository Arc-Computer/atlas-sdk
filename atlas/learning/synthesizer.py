"""Learning synthesizer that maintains persistent pamphlets across sessions."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict

from atlas.config.models import LearningConfig, LLMParameters
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
        pamphlets = {
            "student_pamphlet": state_payload.get("student_learning") if isinstance(state_payload, dict) else None,
            "teacher_pamphlet": state_payload.get("teacher_learning") if isinstance(state_payload, dict) else None,
        }
        payload: Dict[str, Any] = {
            "pamphlets": pamphlets,
            "latest_session": latest_session,
        }
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
        session_student = self._clean_str(payload.get("session_student_learning"))
        session_teacher = self._clean_str(payload.get("session_teacher_learning"))
        updated_student = payload.get("student_pamphlet")
        updated_teacher = payload.get("teacher_pamphlet")
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None

        current_student = baseline_state.get("student_learning") if isinstance(baseline_state, dict) else None
        current_teacher = baseline_state.get("teacher_learning") if isinstance(baseline_state, dict) else None
        current_metadata = baseline_state.get("metadata") if isinstance(baseline_state, dict) else {}

        student_pamphlet = self._clean_str(updated_student)
        teacher_pamphlet = self._clean_str(updated_teacher)
        if student_pamphlet is None:
            student_pamphlet = current_student or ""
        if teacher_pamphlet is None:
            teacher_pamphlet = current_teacher

        note_metadata = metadata if isinstance(metadata, dict) else current_metadata or {}
        session_note = None
        if session_student or session_teacher:
            parts = []
            if session_student:
                parts.append(f"Student: {session_student}")
            if session_teacher:
                parts.append(f"Teacher: {session_teacher}")
            session_note = " ".join(parts)
        learning_state = {
            "student_learning": student_pamphlet,
            "teacher_learning": teacher_pamphlet,
            "metadata": note_metadata,
        }
        return LearningSynthesisResult(
            student_learning=session_student,
            teacher_learning=session_teacher,
            learning_state=learning_state,
            session_note=session_note,
        )

    @staticmethod
    def _clean_str(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value.strip()
        return str(value).strip()


__all__ = ["LearningSynthesizer", "LearningSynthesisResult"]
