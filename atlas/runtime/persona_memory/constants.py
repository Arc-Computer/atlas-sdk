"""Canonical persona identifiers and helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

CANONICAL_PERSONAS: tuple[str, ...] = ("student", "teacher_plan_review", "teacher_validation")

_PERSONA_ALIASES: Dict[str, set[str]] = {
    "student": {"student", "student_planner", "student_executor", "student_synthesizer"},
    "teacher_plan_review": {"teacher_plan_review"},
    "teacher_validation": {"teacher_validation", "teacher_guidance"},
}

_ALIAS_TO_CANONICAL: Dict[str, str] = {}
for canonical, aliases in _PERSONA_ALIASES.items():
    for alias in aliases:
        _ALIAS_TO_CANONICAL[alias] = canonical


def canonical_persona_name(name: str | None) -> str:
    if not name:
        return "student"
    lowered = name.strip()
    if not lowered:
        return "student"
    return _ALIAS_TO_CANONICAL.get(lowered, lowered)


def persona_aliases(persona: str) -> set[str]:
    canonical = canonical_persona_name(persona)
    return set(_PERSONA_ALIASES.get(canonical, {canonical}))


def expand_persona_records(records: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    aggregated: Dict[str, List[dict]] = defaultdict(list)
    for key, payload in records.items():
        canonical = canonical_persona_name(key)
        aggregated[canonical].extend(payload)
    return aggregated

