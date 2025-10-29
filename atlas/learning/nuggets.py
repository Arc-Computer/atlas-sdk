"""Policy nugget schema utilities, gating, and rubric scoring."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from atlas.config.models import PolicyGateRules, PolicyRubricWeights, PolicySchemaConfig

_DATE_PATTERN = re.compile(
    r"\b(?:20\d{2}|19\d{2})(?:[-/](?:0?[1-9]|1[0-2])(?:[-/](?:0?[1-9]|[12]\d|3[01]))?)?\b"
)
_INCIDENT_PATTERN = re.compile(
    r"\b(?:incident|case|ticket)[\s#_\-]+[A-Za-z0-9][A-Za-z0-9_\-]*",
    re.IGNORECASE,
)
_HASH = hashlib.sha256


@dataclass
class NuggetEvaluation:
    """Rubric and gate results for a single policy nugget."""

    gates: Dict[str, bool]
    scores: Dict[str, float]
    weighted_total: float
    failure_reasons: List[str] = field(default_factory=list)

    def passed(self) -> bool:
        return all(self.gates.values())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gates": dict(self.gates),
            "scores": dict(self.scores),
            "weighted_total": float(self.weighted_total),
            "failure_reasons": list(self.failure_reasons),
        }


@dataclass
class EvaluatedNugget:
    """Wrapper struct pairing a normalised nugget with evaluation results."""

    nugget: Dict[str, Any]
    evaluation: NuggetEvaluation

    def passed(self) -> bool:
        return self.evaluation.passed()

    def to_metadata(self) -> Dict[str, Any]:
        payload = dict(self.nugget)
        payload["rubric"] = self.evaluation.to_dict()
        payload.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
        return payload


def normalise_candidates(
    raw_candidates: Iterable[Dict[str, Any] | str],
    *,
    schema: PolicySchemaConfig,
    default_audience: str,
) -> List[Dict[str, Any]]:
    """Coerce raw LLM payloads into structured nugget dictionaries."""

    candidates: List[Dict[str, Any]] = []
    for index, raw in enumerate(raw_candidates or []):
        nugget = _coerce_nugget(raw, schema=schema, default_audience=default_audience, sequence=index)
        candidates.append(nugget)
    return candidates


def evaluate_candidates(
    candidates: Sequence[Dict[str, Any]],
    *,
    gates: PolicyGateRules,
    weights: PolicyRubricWeights,
    schema: PolicySchemaConfig,
    allowed_handles: Sequence[str],
    allowed_prefixes: Sequence[str],
) -> Tuple[List[EvaluatedNugget], Dict[str, Any]]:
    """Evaluate each candidate nugget against rubric gates and weights."""

    allowed_set = {handle.strip() for handle in allowed_handles if handle}
    prefix_set = {prefix.strip() for prefix in allowed_prefixes if prefix}
    evaluations: List[EvaluatedNugget] = []
    gate_failure_totals: Dict[str, int] = {"actionability": 0, "cue": 0, "generality": 0}
    weighted_scores: List[float] = []

    for nugget in candidates:
        evaluation = _evaluate_nugget(
            nugget,
            gates=gates,
            weights=weights,
            schema=schema,
            allowed_handles=allowed_set,
            allowed_prefixes=prefix_set,
        )
        evaluations.append(EvaluatedNugget(nugget=nugget, evaluation=evaluation))
        for gate_name, gate_passed in evaluation.gates.items():
            if not gate_passed:
                gate_failure_totals[gate_name] = gate_failure_totals.get(gate_name, 0) + 1
        weighted_scores.append(evaluation.weighted_total)

    total = len(evaluations)
    passed = sum(1 for item in evaluations if item.passed())
    summary = {
        "total_candidates": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": (passed / total) if total else 0.0,
        "gate_failures": gate_failure_totals,
        "average_weighted_score": (sum(weighted_scores) / len(weighted_scores)) if weighted_scores else 0.0,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return evaluations, summary


def stabilise_nugget_id(nugget: Dict[str, Any]) -> str:
    """Derive a stable identifier for the nugget."""

    parts = [
        nugget.get("audience") or "",
        nugget.get("cue", {}).get("type") or "",
        nugget.get("cue", {}).get("pattern") or nugget.get("cue", {}).get("text") or "",
        nugget.get("action", {}).get("imperative") or "",
        nugget.get("action", {}).get("runtime_handle") or "",
        nugget.get("scope", {}).get("constraints") or "",
    ]
    digest = _HASH("||".join(parts).encode("utf-8")).hexdigest()
    return f"nugget-{digest[:12]}"


def _coerce_nugget(
    raw: Dict[str, Any] | str,
    *,
    schema: PolicySchemaConfig,
    default_audience: str,
    sequence: int,
) -> Dict[str, Any]:
    if isinstance(raw, str):
        payload: Dict[str, Any] = {"action": {"imperative": raw}}
    elif isinstance(raw, dict):
        payload = dict(raw)
    else:
        payload = {"action": {"imperative": str(raw)}}

    payload.setdefault("audience", payload.get("audience", default_audience) or default_audience)
    payload["cue"] = _coerce_cue(payload.get("cue"), schema)
    payload["action"] = _coerce_action(payload.get("action"), schema)
    payload["expected_effect"] = _normalise_string(
        payload.pop("expected_effect", payload.pop("why", "")) or ""
    )
    payload["scope"] = _coerce_scope(payload.get("scope"), schema)
    metadata = payload.get("metadata")
    payload["metadata"] = metadata if isinstance(metadata, dict) else {}
    runtime_mapping = payload.get("runtime_mapping")
    if isinstance(runtime_mapping, dict):
        if payload["action"].get("runtime_handle") is None:
            payload["action"]["runtime_handle"] = runtime_mapping.get("runtime_handle")
        payload["metadata"].setdefault("runtime_mapping", runtime_mapping)
    payload.setdefault("sequence", sequence)
    payload.setdefault("id", payload.get("id") or stabilise_nugget_id(payload))
    return payload


def _coerce_cue(cue: Any, schema: PolicySchemaConfig) -> Dict[str, Any]:
    if isinstance(cue, dict):
        payload = dict(cue)
    elif isinstance(cue, str):
        payload = {"type": "keyword", "pattern": cue}
    else:
        payload = {"type": "keyword", "pattern": ""}
    cue_type = str(payload.get("type") or "").lower()
    if cue_type not in schema.cue_types:
        cue_type = "keyword"
    payload["type"] = cue_type
    pattern = payload.get("pattern") or payload.get("regex") or ""
    if not isinstance(pattern, str):
        pattern = str(pattern)
    payload["pattern"] = pattern.strip()
    description = payload.get("description") or payload.get("text") or ""
    payload["description"] = _normalise_string(description)
    return payload


def _coerce_action(action: Any, schema: PolicySchemaConfig) -> Dict[str, Any]:
    if isinstance(action, dict):
        payload = dict(action)
    elif isinstance(action, str):
        payload = {"imperative": action}
    else:
        payload = {"imperative": ""}
    payload["imperative"] = _normalise_string(payload.get("imperative"))
    handle = payload.get("runtime_handle") or payload.get("handle") or payload.get("tool")
    if not isinstance(handle, str):
        handle = "" if handle is None else str(handle)
    payload["runtime_handle"] = handle.strip() or None
    tool_name = payload.get("tool_name") or payload.get("tool")
    if not isinstance(tool_name, str):
        tool_name = "" if tool_name is None else str(tool_name)
    payload["tool_name"] = tool_name.strip() or None
    arguments = payload.get("arguments") or payload.get("params") or payload.get("parameters")
    if isinstance(arguments, dict):
        payload["arguments"] = dict(arguments)
    else:
        payload["arguments"] = {}
    return payload


def _coerce_scope(scope: Any, schema: PolicySchemaConfig) -> Dict[str, Any]:
    if isinstance(scope, dict):
        payload = dict(scope)
    else:
        text = scope if isinstance(scope, str) else ""
        payload = {"constraints": text}
    payload["constraints"] = _normalise_string(payload.get("constraints"))
    category = payload.get("category") or payload.get("kind")
    if not isinstance(category, str) or not category.strip():
        category = schema.default_scope_category
    payload["category"] = category.strip().lower()
    applies = payload.get("applies_when") or payload.get("applies_to") or ""
    payload["applies_when"] = _normalise_string(applies)
    return payload


def _evaluate_nugget(
    nugget: Dict[str, Any],
    *,
    gates: PolicyGateRules,
    weights: PolicyRubricWeights,
    schema: PolicySchemaConfig,
    allowed_handles: Iterable[str],
    allowed_prefixes: Iterable[str],
) -> NuggetEvaluation:
    gate_results: Dict[str, bool] = {}
    failure_reasons: List[str] = []

    handle = nugget.get("action", {}).get("runtime_handle")
    imperative = nugget.get("action", {}).get("imperative")
    handle_valid = _handle_is_allowed(handle, allowed_handles, allowed_prefixes)
    action_gate = bool(imperative) and (handle_valid or schema.allow_missing_tool_mapping)
    gate_results["actionability"] = (not gates.enforce_actionability) or action_gate
    if gates.enforce_actionability and not action_gate:
        message = "runtime handle missing" if not handle_valid else "imperative missing"
        if not imperative:
            message = "imperative missing"
        failure_reasons.append(f"actionability_gate:{message}")

    cue_payload = nugget.get("cue", {})
    cue_valid = _cue_is_valid(cue_payload, schema)
    gate_results["cue"] = (not gates.enforce_cue) or cue_valid
    if gates.enforce_cue and not cue_valid:
        failure_reasons.append("cue_gate:invalid cue pattern")

    generality_passed, generality_reason = _generality_gate(
        nugget,
        gates=gates,
    )
    gate_results["generality"] = (not gates.enforce_generality) or generality_passed
    if gates.enforce_generality and not generality_passed:
        failure_reasons.append(f"generality_gate:{generality_reason}")

    scores = _compute_scores(
        nugget,
        weights=weights,
        gates=gates,
        gate_results=gate_results,
        cue_valid=cue_valid,
    )
    weights_map = _normalised_weights(weights)
    weighted_total = sum(scores[name] * weights_map[name] for name in weights_map)

    return NuggetEvaluation(
        gates=gate_results,
        scores=scores,
        weighted_total=weighted_total,
        failure_reasons=failure_reasons,
    )


def _compute_scores(
    nugget: Dict[str, Any],
    *,
    weights: PolicyRubricWeights,
    gates: PolicyGateRules,
    gate_results: Dict[str, bool],
    cue_valid: bool,
) -> Dict[str, float]:
    actionability_score = 1.0 if gate_results.get("actionability", True) else 0.0
    if gates.enforce_actionability and not gate_results.get("actionability", False):
        actionability_score = 0.0

    generality_score = 1.0 if gate_results.get("generality", True) else 0.0

    cue = nugget.get("cue", {})
    cue_type = (cue.get("type") or "").lower()
    hookability_score = 1.0 if cue_type == "regex" else 0.6 if cue_type == "predicate" else 0.4
    if not cue_valid:
        hookability_score = 0.0

    combined = " ".join(
        segment
        for segment in (
            nugget.get("action", {}).get("imperative"),
            nugget.get("expected_effect"),
            nugget.get("scope", {}).get("constraints"),
            nugget.get("cue", {}).get("pattern"),
        )
        if isinstance(segment, str)
    )
    text_length = len(combined)
    length_limit = gates.max_text_length
    concision_score = 1.0
    if text_length > length_limit:
        overflow = text_length - length_limit
        concision_score = max(0.0, 1.0 - (overflow / max(length_limit, 1)))

    return {
        "actionability": round(actionability_score, 4),
        "generality": round(generality_score, 4),
        "hookability": round(hookability_score, 4),
        "concision": round(concision_score, 4),
    }


def _handle_is_allowed(handle: str | None, allowed_handles: Iterable[str], allowed_prefixes: Iterable[str]) -> bool:
    if not handle:
        return False
    handle = handle.strip()
    if not handle:
        return False
    lower_handle = handle.lower()
    allowed = {candidate.lower() for candidate in allowed_handles if candidate}
    if lower_handle in allowed:
        return True
    for prefix in allowed_prefixes:
        if prefix and lower_handle.startswith(prefix.lower()):
            return True
    return False


def _cue_is_valid(cue: Dict[str, Any], schema: PolicySchemaConfig) -> bool:
    cue_type = (cue.get("type") or "").lower()
    if cue_type not in schema.cue_types:
        return False
    pattern = cue.get("pattern") or cue.get("text") or ""
    if not isinstance(pattern, str) or not pattern.strip():
        return False
    if cue_type == "regex":
        try:
            re.compile(pattern)
        except re.error:
            return False
    return True


def _generality_gate(
    nugget: Dict[str, Any],
    *,
    gates: PolicyGateRules,
) -> Tuple[bool, str]:
    text_segments: List[str] = []
    cue = nugget.get("cue", {})
    action = nugget.get("action", {})
    scope = nugget.get("scope", {})
    analysis_segments: List[str] = []
    text_segments.extend(
        segment
        for segment in (
            cue.get("pattern"),
            cue.get("description"),
            action.get("imperative"),
            nugget.get("expected_effect"),
            scope.get("constraints"),
            scope.get("applies_when"),
        )
        if isinstance(segment, str)
    )
    for segment in text_segments:
        words = segment.split()
        if len(words) > 1:
            analysis_segments.append(" ".join(words[1:]))
    combined = " ".join(text_segments)
    if not combined:
        return False, "empty_text"
    lower = combined.lower()
    for token in gates.banned_incident_tokens:
        if token and token in lower:
            return False, f"banned_token:{token}"
    if _DATE_PATTERN.search(combined):
        return False, "contains_date"
    if _INCIDENT_PATTERN.search(combined):
        return False, "contains_incident_reference"
    analysis_text = " ".join(analysis_segments) if analysis_segments else combined
    offending = _detect_proper_nouns(analysis_text, gates.allowed_proper_nouns)
    if offending:
        return False, f"proper_nouns:{','.join(offending[:3])}"
    if len(combined) > (gates.max_text_length + gates.allow_length_overflow_margin):
        return False, "exceeds_length"
    return True, ""


def _detect_proper_nouns(text: str, allowlist: Sequence[str]) -> List[str]:
    allow = {entry.lower() for entry in allowlist}
    offenders: List[str] = []
    for match in re.finditer(r"\b[A-Z][a-z]{2,}\b", text):
        token = match.group(0)
        start = match.start()
        if start == 0:
            continue
        preceding = text[start - 1]
        if preceding in {".", "!", "?", "\n"}:
            continue
        if token.lower() not in allow:
            offenders.append(token)
    return offenders


def _normalise_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _normalised_weights(weights: PolicyRubricWeights) -> Dict[str, float]:
    raw = {
        "actionability": max(weights.actionability, 0.0),
        "generality": max(weights.generality, 0.0),
        "hookability": max(weights.hookability, 0.0),
        "concision": max(weights.concision, 0.0),
    }
    total = sum(raw.values()) or 1.0
    return {key: round(value / total, 4) for key, value in raw.items()}


__all__ = [
    "EvaluatedNugget",
    "NuggetEvaluation",
    "evaluate_candidates",
    "normalise_candidates",
    "stabilise_nugget_id",
]
