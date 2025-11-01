"""LLM-backed capability probe used to select adaptive execution modes."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from atlas.config.models import AdaptiveProbeConfig, LLMParameters, LLMProvider
from atlas.utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

_DEFAULT_PROBE_LLM = LLMParameters(
    provider=LLMProvider.XAI,
    model="xai/grok-4-fast",
    api_key_env="XAI_API_KEY",
    temperature=0.2,
    timeout_seconds=20.0,
)

_SYSTEM_PROMPT = """Role: Adaptive execution router.

Decide which runtime mode should execute the next request based on the task description and accumulated learning history and reward scores.

Candidate modes:
- Use `auto` when the learning history shows high scores on similar tasks and no recent regressions.
- auto: student runs unattended (fast lane).
- Use `paired` when the student has helpful learning but low reward score, so still needs a teacher review to confirm the final answer.
- paired: student runs once with teacher inspecting final answer.
- Use `coach` when the task is partially familiar but low reward score, so needs plan validation and targeted guidance.
- coach: teacher reviews the plan and check the final answer.

Requirements:
1. Review the learning history (reward scores, student/teacher learnings) to judge how confident we should be.
2. Consider how similar tasks were handled previously when choosing a mode.
3. Return a single JSON object with this exact schema:
   {{
     "mode": "auto" | "paired" | "coach",
     "confidence": float | null,
   }}
4. Confidence must be between 0 and 1 when provided.
"""


@dataclass(slots=True)
class CapabilityProbeDecision:
    mode: Optional[str]
    confidence: Optional[float]
    raw: Dict[str, Any] | None = None


class CapabilityProbeClient:
    """Thin wrapper around an LLM that selects adaptive execution modes.

    Auto-detects missing API credentials and gracefully disables probe,
    falling back to deterministic mode selection.
    """

    def __init__(self, config: AdaptiveProbeConfig | None) -> None:
        self._config = config or AdaptiveProbeConfig()
        llm_params = self._config.llm or _DEFAULT_PROBE_LLM
        self._fallback_mode = self._config.fallback_mode
        self._timeout = self._config.timeout_seconds

        # Auto-detect if probe can be enabled based on API key availability
        api_key = os.getenv(llm_params.api_key_env)
        if not api_key:
            logger.warning(
                "Capability probe disabled: API key '%s' not found. "
                "Falling back to '%s' mode. To enable probe, set the environment variable.",
                llm_params.api_key_env,
                self._fallback_mode,
            )
            self._enabled = False
            self._client = None
        else:
            self._enabled = True
            self._client = LLMClient(llm_params)

    async def arun(
        self,
        *,
        task: str,
        dossier: Dict[str, Any],
        execution_metadata: Dict[str, Any],
    ) -> CapabilityProbeDecision:
        if not self._enabled:
            return CapabilityProbeDecision(
                mode=None,
                confidence=None,
                raw={"disabled": True, "reason": "Missing API credentials"},
            )

        payload = self._build_payload(task, dossier, execution_metadata)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        overrides = {"timeout": self._timeout} if self._timeout else None
        response = await self._client.acomplete(
            messages,
            response_format={"type": "json_object"},
            overrides=overrides,
        )
        decision = self._parse_response(response.content)
        decision.raw = self._safe_json_loads(response.content)
        return decision

    def _build_payload(
        self,
        task: str,
        dossier: Dict[str, Any],
        execution_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        _ = dossier  # intentionally unused; learning history drives routing
        learning_history = execution_metadata.get("learning_history")
        payload: Dict[str, Any] = {
            "task": task,
            "learning_history": learning_history,
        }
        return payload

    def _parse_response(self, content: str) -> CapabilityProbeDecision:
        data = self._safe_json_loads(content)
        if not isinstance(data, dict):
            return CapabilityProbeDecision(
                mode=None,
                confidence=None,
                raw=None,
            )
        mode = data.get("mode")
        confidence = data.get("confidence")
        numeric_confidence = None
        if isinstance(confidence, (int, float)):
            numeric_confidence = float(confidence)
        return CapabilityProbeDecision(
            mode=str(mode) if isinstance(mode, str) else None,
            confidence=numeric_confidence,
            raw=data,
        )

    @staticmethod
    def _safe_json_loads(payload: str) -> Dict[str, Any] | Any:
        try:
            return json.loads(payload)
        except (TypeError, ValueError):
            return payload

    @property
    def fallback_mode(self) -> str:
        return self._fallback_mode if self._fallback_mode in {"paired", "coach"} else "paired"
