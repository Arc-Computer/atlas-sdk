"""Persona memory runtime helpers."""

from __future__ import annotations

from .cache import PersonaMemoryCache, PersonaMemoryKey, get_cache, is_cache_disabled
from .constants import (
    CANONICAL_PERSONAS,
    canonical_persona_name,
    expand_persona_records,
    persona_aliases,
)
from .fingerprint import FingerprintInputs, build_fingerprint, extract_fingerprint_inputs
from .learning import CandidateSpec, extract_candidates, write_candidates
from .merge import PersonaMemoryInstruction, merge_prompt, normalize_instructions
from .promotion import PromotionResult, PromotionSettings, get_promotion_settings, promote_and_compact

__all__ = [
    "FingerprintInputs",
    "build_fingerprint",
    "extract_fingerprint_inputs",
    "PersonaMemoryCache",
    "PersonaMemoryKey",
    "get_cache",
    "is_cache_disabled",
    "PersonaMemoryInstruction",
    "merge_prompt",
    "normalize_instructions",
    "CandidateSpec",
    "extract_candidates",
    "write_candidates",
    "PromotionSettings",
    "PromotionResult",
    "get_promotion_settings",
    "promote_and_compact",
    "CANONICAL_PERSONAS",
    "canonical_persona_name",
    "persona_aliases",
    "expand_persona_records",
]
