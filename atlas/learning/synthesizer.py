from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from atlas.config.models import LearningConfig
from atlas.evaluation.evaluator import SessionTrajectory
from atlas.learning.prompts import LEARNING_SYNTHESIS_PROMPT
from atlas.runtime.schema import AtlasRewardBreakdown
from atlas.runtime.storage.database import Database
from atlas.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LearningUpdate:
    session_student_learning: Optional[str]
    session_teacher_learning: Optional[str]
    updated_state: Optional[Dict[str, Any]]
    history_snapshot: Optional[Dict[str, Any]]
    session_metadata: Optional[Dict[str, Any]] = None


class LearningSynthesizer:
    def __init__(self, config: LearningConfig, database: Database | None = None) -> None:
        self._config = config
        self._prompt = config.prompt or LEARNING_SYNTHESIS_PROMPT
        self._client = LLMClient(config.llm)
        self._database = database
        self._max_history = max(0, config.max_history_entries)

    @property
    def updates_enabled(self) -> bool:
        return bool(self._config.updates_enabled)

    async def asynthesize(
        self,
        *,
        learning_key: str,
        trajectory: SessionTrajectory,
        reward: AtlasRewardBreakdown,
        history_snapshot: Dict[str, Any] | None,
        prior_state: Dict[str, Any] | None,
    ) -> LearningUpdate | None:
        student_doc = self._as_text(prior_state, "student_learning")
        teacher_doc = self._as_text(prior_state, "teacher_learning")
        if not self.updates_enabled:
            state_payload = prior_state if isinstance(prior_state, dict) else {
                "learning_key": learning_key,
                "student_learning": student_doc,
                "teacher_learning": teacher_doc,
                "metadata": None,
                "updated_at": None,
            }
            return LearningUpdate(
                session_student_learning=student_doc,
                session_teacher_learning=teacher_doc,
                updated_state=state_payload,
                history_snapshot=self._merge_history_snapshot(history_snapshot, state_payload),
            )
        payload = {
            "learning_key": learning_key,
            "current_pamphlet": {
                "student": student_doc,
                "teacher": teacher_doc,
                "metadata": prior_state.get("metadata") if isinstance(prior_state, dict) else None,
                "updated_at": prior_state.get("updated_at") if isinstance(prior_state, dict) else None,
            },
            "session": self._build_session_snapshot(trajectory, reward),
            "history": self._prepare_history(history_snapshot),
        }
        messages = [
            {"role": "system", "content": self._prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=self._json_default)},
        ]
        try:
            response = await self._client.acomplete(
                messages,
                response_format={"type": "json_object"},
            )
            parsed = self._safe_json_loads(response.content)
        except Exception as exc:
            logger.exception("Learning synthesizer call failed for key %s: %s", learning_key, exc)
            state_payload = prior_state if isinstance(prior_state, dict) else {
                "learning_key": learning_key,
                "student_learning": student_doc,
                "teacher_learning": teacher_doc,
                "metadata": None,
                "updated_at": None,
            }
            return LearningUpdate(
                session_student_learning=student_doc,
                session_teacher_learning=teacher_doc,
                updated_state=state_payload,
                history_snapshot=self._merge_history_snapshot(history_snapshot, state_payload),
            )
        if not isinstance(parsed, dict):
            logger.warning("Learning synthesizer returned non-dict payload for key %s", learning_key)
            state_payload = prior_state if isinstance(prior_state, dict) else {
                "learning_key": learning_key,
                "student_learning": student_doc,
                "teacher_learning": teacher_doc,
                "metadata": None,
                "updated_at": None,
            }
            return LearningUpdate(
                session_student_learning=student_doc,
                session_teacher_learning=teacher_doc,
                updated_state=state_payload,
                history_snapshot=self._merge_history_snapshot(history_snapshot, state_payload),
            )
        update = self._normalise_payload(parsed, student_doc, teacher_doc)
        stored_state = await self._persist_state(
            learning_key,
            update["student_pamphlet"],
            update["teacher_pamphlet"],
            update["metadata"],
        )
        history_payload = self._merge_history_snapshot(history_snapshot, stored_state)
        return LearningUpdate(
            session_student_learning=update["session_student_learning"],
            session_teacher_learning=update["session_teacher_learning"],
            updated_state=stored_state,
            history_snapshot=history_payload,
            session_metadata=update["metadata"] if isinstance(update["metadata"], dict) else None,
        )

    async def _persist_state(
        self,
        learning_key: str,
        student: str | None,
        teacher: str | None,
        metadata: Any,
    ) -> Dict[str, Any] | None:
        if self._database is None:
            return {
                "learning_key": learning_key,
                "student_learning": student,
                "teacher_learning": teacher,
                "metadata": metadata if isinstance(metadata, dict) else None,
                "updated_at": None,
            }
        await self._database.upsert_learning_state(learning_key, student, teacher, metadata if isinstance(metadata, dict) else None)
        stored = await self._database.fetch_learning_state(learning_key)
        if stored is None:
            return {
                "learning_key": learning_key,
                "student_learning": student,
                "teacher_learning": teacher,
                "metadata": metadata if isinstance(metadata, dict) else None,
                "updated_at": None,
            }
        return stored

    def _prepare_history(self, snapshot: Dict[str, Any] | None) -> List[Dict[str, Any]]:
        if not isinstance(snapshot, dict):
            return []
        entries = snapshot.get("entries")
        if not isinstance(entries, list):
            return []
        if self._max_history > 0:
            entries = entries[-self._max_history :]
        trimmed: List[Dict[str, Any]] = []
        for entry in entries:
            if isinstance(entry, dict):
                trimmed.append(
                    {
                        "reward": entry.get("reward"),
                        "student_learning": entry.get("student_learning"),
                        "teacher_learning": entry.get("teacher_learning"),
                        "created_at": entry.get("created_at"),
                        "completed_at": entry.get("completed_at"),
                    }
                )
        return trimmed

    def _build_session_snapshot(
        self,
        trajectory: SessionTrajectory,
        reward: AtlasRewardBreakdown,
    ) -> Dict[str, Any]:
        judge = reward.judges[0] if reward.judges else None
        metadata = trajectory.session_metadata or {}
        evaluation = metadata.get("secrl_evaluation") if isinstance(metadata, dict) else None
        canonical_solution = metadata.get("secrl_solution") if isinstance(metadata, dict) else None
        student_submission = metadata.get("secrl_final_answer") if isinstance(metadata, dict) else None
        teacher_guidance = metadata.get("secrl_applied_guidance") if isinstance(metadata, dict) else None
        reward_payload = reward.to_dict() if hasattr(reward, "to_dict") else None
        return {
            "task": trajectory.task,
            "execution_mode": trajectory.execution_mode,
            "teacher_intervened": trajectory.teacher_intervened,
            "final_answer": trajectory.final_answer,
            "reward_score": reward.score,
            "reward_rationale": judge.rationale if judge else None,
            "focus_prompt": trajectory.focus_prompt,
            "student_submission": student_submission,
            "canonical_solution": canonical_solution,
            "judge_feedback": evaluation,
            "teacher_guidance": teacher_guidance,
            "reward_breakdown": reward_payload,
            "session_metadata": metadata,
        }

    def _normalise_payload(
        self,
        payload: Dict[str, Any],
        fallback_student: str | None,
        fallback_teacher: str | None,
    ) -> Dict[str, Any]:
        student = self._coerce_text(payload.get("student_pamphlet")) or fallback_student
        teacher = self._coerce_text(payload.get("teacher_pamphlet")) or fallback_teacher
        session_student = self._coerce_text(payload.get("session_student_learning")) or student
        session_teacher = self._coerce_text(payload.get("session_teacher_learning")) or teacher
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = None
        return {
            "student_pamphlet": student,
            "teacher_pamphlet": teacher,
            "session_student_learning": session_student,
            "session_teacher_learning": session_teacher,
            "metadata": metadata,
        }

    @staticmethod
    def _safe_json_loads(payload: str) -> Any:
        try:
            return json.loads(payload)
        except (TypeError, ValueError):
            return payload

    @staticmethod
    def _coerce_text(value: Any) -> str | None:
        if isinstance(value, str):
            text = value.strip()
            return text if text else None
        return None

    @staticmethod
    def _as_text(payload: Dict[str, Any] | None, key: str) -> str | None:
        if isinstance(payload, dict):
            value = payload.get(key)
            if isinstance(value, str):
                text = value.strip()
                if text:
                    return text
        return None

    def _merge_history_snapshot(
        self,
        snapshot: Dict[str, Any] | None,
        state: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        base = dict(snapshot) if isinstance(snapshot, dict) else {}
        if isinstance(state, dict):
            student = state.get("student_learning")
            teacher = state.get("teacher_learning")
            metadata = state.get("metadata")
            timestamp = state.get("updated_at")
            if isinstance(student, str) and student:
                base["current_student_learning"] = student
            if isinstance(teacher, str) and teacher:
                base["current_teacher_learning"] = teacher
            if isinstance(metadata, dict) and metadata:
                base["current_metadata"] = metadata
            iso_timestamp = self._timestamp_to_iso(timestamp)
            if iso_timestamp:
                base["current_updated_at"] = iso_timestamp
        entries = base.get("entries")
        if not isinstance(entries, list):
            base["entries"] = []
        if "count" not in base or not isinstance(base["count"], int):
            base["count"] = len(base["entries"])
        return base

    @staticmethod
    def _timestamp_to_iso(value: Any) -> str | None:
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception:  # pragma: no cover - defensive guard
                return None
        if isinstance(value, str):
            return value
        return None

    @staticmethod
    def _json_default(value: Any) -> Any:
        if hasattr(value, "isoformat"):
            try:
                return value.isoformat()
            except Exception:  # pragma: no cover
                return str(value)
        return str(value)
