"""BYOA agent for the streaming SRE demo leveraging real OpenAI completions."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict

import litellm

SYSTEM_PROMPT = (
    "You are Atlas' incident diagnostics agent. You receive payloads describing an SRE incident. "
    "Extract the likely root cause, mention any key signals, and provide a confidence score between 0.0 and 1.0. "
    "Respond as JSON with fields: root_cause (snake_case), confidence (float), and rationale (concise string). "
    "Stay focused on production impact and avoid remediation steps."
)

DEFAULT_MODEL = os.getenv("SRE_AGENT_MODEL", "gpt-5")
DEFAULT_TEMPERATURE = float(os.getenv("SRE_AGENT_TEMPERATURE", "1.0"))


def _extract_json_block(text: str) -> dict[str, Any]:
    if not text:
        return {}
    start = text.find("{")
    while start != -1:
        end = text.rfind("}")
        while end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                end = text.rfind("}", start, end)
        start = text.find("{", start + 1)
    return {}


def _safe_parse_prompt(prompt: str) -> dict[str, Any]:
    try:
        data = json.loads(prompt)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        return _extract_json_block(prompt)
    return {}


def _fallback_from_triage(triage_payload: dict[str, Any]) -> dict[str, Any]:
    overview = triage_payload.get("overview") or {}
    incident_type = overview.get("incident_type") or "insufficient_evidence"
    summary = triage_payload.get("summary") or "No summary available."
    runbook_hint = triage_payload.get("runbook_context")
    rationale_parts = []
    if runbook_hint:
        rationale_parts.append(runbook_hint)
    rationale_parts.append(f"Triage summary: {summary}")
    rationale = " ".join(rationale_parts)
    return {
        "root_cause": incident_type or "insufficient_evidence",
        "confidence": 0.25 if runbook_hint else 0.15,
        "rationale": rationale,
    }


async def _acomplete(messages: list[dict[str, Any]], *, api_key: str, model: str) -> str:
    temperature = DEFAULT_TEMPERATURE
    if model.startswith("gpt-5") and temperature != 1.0:
        temperature = 1.0
    result = await litellm.acompletion(
        model=model,
        messages=messages,
        api_key=api_key,
        temperature=temperature,
        max_tokens=512,
        response_format={"type": "json_object"},
    )
    try:
        if hasattr(result, "choices"):
            choice = result.choices[0]
            message = getattr(choice, "message", {}) or {}
            provider_fields = getattr(message, "provider_specific_fields", {}) or {}
            tool_calls = getattr(message, "tool_calls", None)
            content = getattr(message, "content", None)
        else:
            choice = result["choices"][0]
            message = choice["message"]
            provider_fields = message.get("provider_specific_fields") or {}
            tool_calls = message.get("tool_calls")
            content = message.get("content")
    except (KeyError, IndexError, TypeError, AttributeError) as exc:  # pragma: no cover - network issues
        raise RuntimeError(f"Unexpected OpenAI response payload: {result}") from exc
    if isinstance(content, list):
        content = "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        ).strip()
    if not content and tool_calls:
        content = json.dumps({"tool_calls": tool_calls}, ensure_ascii=False)
    if not content and provider_fields:
        refusal = provider_fields.get("refusal")
        if refusal:
            content = json.dumps({"refusal": refusal}, ensure_ascii=False)
    if not content:
        content = json.dumps(
            {
                "root_cause": "unknown",
                "confidence": 0.05,
                "rationale": "Model returned empty response.",
            }
        )
    return str(content)


def _normalise_response(raw: str) -> str:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            payload = {
                "root_cause": parsed.get("root_cause") or parsed.get("diagnosis") or "unknown",
                "confidence": float(parsed.get("confidence", 0.2)),
                "rationale": parsed.get("rationale") or parsed.get("explanation") or "",
            }
        else:
            raise ValueError("Response was not JSON object")
    except (json.JSONDecodeError, ValueError):
        payload = {
            "root_cause": "unknown",
            "confidence": 0.15,
            "rationale": raw.strip(),
        }
    if not (0.0 <= payload["confidence"] <= 1.0):
        payload["confidence"] = max(0.0, min(1.0, float(payload["confidence"] or 0.15)))
    return json.dumps(payload)


async def diagnose_incident(prompt: str, metadata: Dict[str, Any] | None = None) -> str:
    """Diagnose an incident using a real OpenAI completion and return JSON output."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is required for the SRE demo agent.")
    model = DEFAULT_MODEL
    if metadata:
        model = metadata.get("preferred_model", model) or model
    triage_payload = _safe_parse_prompt(prompt)
    mode = metadata.get("mode") if metadata else None
    if mode == "planning":
        steps = []
        summary = triage_payload.get("summary", "Review incident context.")
        signals = triage_payload.get("signals", {})
        recent_changes = signals.get("recent_changes") or []
        change_summary = recent_changes[0]["summary"] if recent_changes else "Check latest deploys for contributing changes."
        runbook_hint = triage_payload.get("runbook_context") or "Consult runbook for remediation guidance."
        steps.append(
            {
                "id": 1,
                "description": f"Review incident summary and metrics: {summary}",
                "tool": None,
                "tool_params": None,
                "depends_on": [],
            }
        )
        steps.append(
            {
                "id": 2,
                "description": f"Correlate with recent change: {change_summary}",
                "tool": None,
                "tool_params": None,
                "depends_on": [1],
            }
        )
        steps.append(
            {
                "id": 3,
                "description": f"Formulate diagnosis and next step based on runbook: {runbook_hint}",
                "tool": None,
                "tool_params": None,
                "depends_on": [2],
            }
        )
        plan = {"execution_mode": "stepwise", "steps": steps}
        return json.dumps(plan)
    if mode == "synthesis":
        summary = triage_payload.get("summary", "Incident summarised.")
        runbook_hint = triage_payload.get("runbook_context")
        result = {
            "summary": summary,
            "next_steps": runbook_hint or "Follow promoted persona guidance to remediate.",
        }
        return json.dumps(result)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    content = await _acomplete(messages, api_key=api_key, model=model)
    normalised = _normalise_response(content)
    try:
        parsed = json.loads(normalised)
    except json.JSONDecodeError:
        parsed = {"root_cause": "unknown", "confidence": 0.15, "rationale": normalised}
    if (parsed.get("root_cause") in (None, "", "unknown") or float(parsed.get("confidence", 0.0)) < 0.2) and triage_payload:
        parsed = _fallback_from_triage(triage_payload)
        normalised = json.dumps(parsed)
    print(f"[Agent] diagnosis completed using {model}")  # Useful during live demos.
    return normalised


def diagnose_incident_sync(prompt: str, metadata: Dict[str, Any] | None = None) -> str:
    """Synchronous adapter for completeness (not used by the demo)."""

    return asyncio.run(diagnose_incident(prompt=prompt, metadata=metadata))


__all__ = ["diagnose_incident", "diagnose_incident_sync"]
