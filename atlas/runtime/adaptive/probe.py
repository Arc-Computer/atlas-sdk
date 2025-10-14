"""LLM-backed capability probe used to select adaptive execution modes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from atlas.config.models import AdaptiveProbeConfig, LLMParameters, LLMProvider
from atlas.utils.llm_client import LLMClient

_DEFAULT_PROBE_LLM = LLMParameters(
    provider=LLMProvider.GOOGLE,
    model="gemini/gemini-2.5-flash",
    api_key_env="GEMINI_API_KEY",
    temperature=0.2,
    timeout_seconds=20.0,
)

_SYSTEM_PROMPT = """Role: Adaptive execution router.

Decide which runtime mode should execute the next request based on the fingerprint and triage summary.

Candidate modes:
- auto: student runs unattended (fast lane).
- paired: student runs once with teacher inspecting final answer.
- coach: teacher reviews the plan and check the final answer.
- escalate: teacher supervises step-by-step with remediation.

Requirements:
1. Analyse fingerprint history and triage dossier risks to determine how confident the system should be.
2. Consider recent adaptive history (previous modes, confidence, outcomes) when available.
3. Return a single JSON object with this exact schema:
   {{
     "mode": "auto" | "paired" | "coach" | "escalate",
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
    """Thin wrapper around an LLM that selects adaptive execution modes."""

    def __init__(self, config: AdaptiveProbeConfig | None) -> None:
        self._config = config or AdaptiveProbeConfig()
        llm_params = self._config.llm or _DEFAULT_PROBE_LLM
        self._client = LLMClient(llm_params)
        self._fallback_mode = self._config.fallback_mode
        self._timeout = self._config.timeout_seconds

    async def arun(
        self,
        *,
        task: str,
        dossier: Dict[str, Any],
        fingerprint: str | None,
        execution_metadata: Dict[str, Any],
    ) -> CapabilityProbeDecision:
        payload = self._build_payload(task, dossier, fingerprint, execution_metadata)
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
        fingerprint: str | None,
        execution_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        adaptive_history = execution_metadata.get("adaptive", {})
        recent_reward = execution_metadata.get("session_reward")
        payload: Dict[str, Any] = {
            "task": task,
            "fingerprint": fingerprint,
            "triage_dossier": dossier,
            "adaptive_history": adaptive_history,
            "recent_reward": recent_reward,
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
        return self._fallback_mode if self._fallback_mode in {"paired", "escalate"} else "paired"
