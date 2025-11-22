"""LLM-driven configuration inference for atlas env init.

This module provides intelligent candidate selection and configuration
recommendations using Anthropic Claude models.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from atlas.sdk.discovery import Candidate

logger = logging.getLogger(__name__)


@dataclass
class CandidateSelection:
    """Result of LLM-based candidate selection."""

    candidate: Candidate
    confidence: float
    reasoning: str
    provider_recommendations: Optional[Dict[str, str]] = None


@dataclass
class ProviderAvailability:
    """Track which LLM providers have API keys available."""

    anthropic: bool = False
    openai: bool = False
    google: bool = False
    bedrock: bool = False
    xai: bool = False

    @property
    def has_any(self) -> bool:
        """Check if any provider is available."""
        return any([self.anthropic, self.openai, self.google, self.bedrock, self.xai])

    @property
    def primary_provider(self) -> Optional[str]:
        """Return the first available provider in priority order."""
        if self.anthropic:
            return "anthropic"
        if self.openai:
            return "openai"
        if self.google:
            return "google"
        if self.bedrock:
            return "bedrock"
        if self.xai:
            return "xai"
        return None


def _detect_available_providers() -> ProviderAvailability:
    """Detect which LLM providers have API keys configured.

    Returns:
        ProviderAvailability with flags for each provider
    """
    return ProviderAvailability(
        anthropic=bool(os.getenv("ANTHROPIC_API_KEY")),
        openai=bool(os.getenv("OPENAI_API_KEY")),
        google=bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
        bedrock=bool(os.getenv("AWS_ACCESS_KEY_ID")),
        xai=bool(os.getenv("XAI_API_KEY")),
    )


def heuristic_rank_candidates(candidates: List[Candidate]) -> Candidate:
    """Fallback heuristic ranking when LLM selection is unavailable.

    Ranking criteria (in priority order):
    1. Decorated candidates (explicit @agent/@environment)
    2. Higher discovery score
    3. More complete capability methods

    Args:
        candidates: List of discovered agent candidates

    Returns:
        Top-ranked candidate
    """
    if not candidates:
        raise ValueError("No candidates provided for ranking")

    if len(candidates) == 1:
        return candidates[0]

    # Sort by: decorator presence (desc), score (desc)
    sorted_candidates = sorted(
        candidates, key=lambda c: (c.via_decorator, c.score), reverse=True
    )

    top = sorted_candidates[0]
    logger.info(
        f"Heuristic selection: {top.dotted_path()} "
        f"(decorator={top.via_decorator}, score={top.score:.2f})"
    )

    return top


def llm_select_candidate(
    candidates: List[Candidate],
    codebase_context: Optional[str] = None,
    api_key: Optional[str] = None,
) -> CandidateSelection:
    """Use Anthropic Claude Haiku to intelligently select the best agent candidate.

    This function analyzes multiple agent candidates and returns the most suitable
    one with reasoning and confidence score. Falls back to heuristic ranking if
    LLM selection fails.

    Args:
        candidates: List of discovered agent candidates
        codebase_context: Optional context about the codebase structure
        api_key: Optional Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)

    Returns:
        CandidateSelection with chosen candidate, confidence, and reasoning
    """
    if not candidates:
        raise ValueError("No candidates provided for selection")

    if len(candidates) == 1:
        logger.info(f"Single candidate found: {candidates[0].dotted_path()}")
        return CandidateSelection(
            candidate=candidates[0],
            confidence=1.0,
            reasoning="Only one agent candidate discovered",
        )

    # Check for Anthropic API key
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        logger.warning(
            "ANTHROPIC_API_KEY not found, falling back to heuristic selection"
        )
        top_candidate = heuristic_rank_candidates(candidates)
        return CandidateSelection(
            candidate=top_candidate,
            confidence=0.7,
            reasoning="Heuristic selection (LLM unavailable): "
            f"{'decorated' if top_candidate.via_decorator else 'discovered'} "
            f"candidate with score {top_candidate.score:.2f}",
        )

    # Build candidate descriptions for LLM
    candidate_descriptions = []
    for idx, candidate in enumerate(candidates, start=1):
        # Extract capability list from capabilities dict
        caps = []
        if candidate.capabilities:
            for method, present in candidate.capabilities.items():
                if present:
                    caps.append(method)

        desc = {
            "index": idx,
            "qualified_name": candidate.dotted_path(),
            "file_path": str(candidate.file_path),
            "discovery_method": "decorator" if candidate.via_decorator else "heuristic",
            "discovery_score": candidate.score,  # Already an int, no need to round
            "capabilities": caps,
        }
        candidate_descriptions.append(desc)

    # Build prompt for LLM
    prompt = _build_selection_prompt(candidate_descriptions, codebase_context)

    # Call Anthropic API
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=key)

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            temperature=0.2,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        # Parse JSON response
        response_text = response.content[0].text

        # Try to extract JSON from markdown code blocks if present
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()

        result = json.loads(response_text)

        selected_index = result["selected_index"]
        confidence = result["confidence"]
        reasoning = result["reasoning"]

        if selected_index < 1 or selected_index > len(candidates):
            raise ValueError(
                f"Invalid selection index: {selected_index} (must be 1-{len(candidates)})"
            )

        selected_candidate = candidates[selected_index - 1]

        logger.info(
            f"LLM selected: {selected_candidate.dotted_path()} "
            f"(confidence={confidence:.2f})"
        )

        return CandidateSelection(
            candidate=selected_candidate,
            confidence=confidence,
            reasoning=reasoning,
        )

    except Exception as e:
        logger.warning(f"LLM selection failed: {e}, falling back to heuristic")
        top_candidate = heuristic_rank_candidates(candidates)
        return CandidateSelection(
            candidate=top_candidate,
            confidence=0.7,
            reasoning=f"Heuristic fallback after LLM error: {str(e)[:100]}",
        )


def _build_selection_prompt(
    candidate_descriptions: List[Dict], codebase_context: Optional[str]
) -> str:
    """Build the LLM prompt for candidate selection.

    Args:
        candidate_descriptions: List of candidate metadata dicts
        codebase_context: Optional additional context about the codebase

    Returns:
        Formatted prompt string
    """
    candidates_json = json.dumps(candidate_descriptions, indent=2)

    context_section = ""
    if codebase_context:
        context_section = f"\n\nCodebase Context:\n{codebase_context}\n"

    prompt = f"""You are helping configure the Atlas SDK for a user's agent codebase.

Atlas SDK is a dual-agent training framework where:
- Student = the user's existing agent (what they want to improve)
- Teacher = validation layer that provides supervision
- The goal is to wrap their existing agent with minimal friction

You have discovered multiple agent candidates in their codebase. Your task is to select the BEST candidate that represents their primary agent implementation.

Candidates:
{candidates_json}
{context_section}
Selection Criteria (in priority order):
1. **Explicit declaration**: Candidates with decorators (@agent, @environment) are explicitly marked
2. **Completeness**: Candidates with more capability methods (step, assess, etc.) are more complete
3. **Discovery score**: Higher scores indicate stronger pattern matches
4. **File location**: Candidates in core agent directories (agents/, src/, lib/) are more likely to be primary agents

Your response must be valid JSON matching this schema:
{{
  "selected_index": <integer 1-{len(candidate_descriptions)}>,
  "confidence": <float 0.0-1.0>,
  "reasoning": "<1-2 sentence explanation of why this candidate was chosen>"
}}

Guidelines:
- Confidence should be 0.9+ if there's a clear decorated candidate
- Confidence should be 0.7-0.9 if heuristic score strongly suggests one candidate
- Confidence should be 0.5-0.7 if candidates are similar (use discovery score as tiebreaker)
- Reasoning should mention the key differentiator (decorator, score, capabilities, location)

Respond with ONLY the JSON object, no additional text."""

    return prompt


def detect_adapter_type(candidate: Candidate) -> str:
    """Detect the appropriate Atlas adapter type for a candidate.

    Args:
        candidate: The selected agent candidate

    Returns:
        Adapter type string: "python", "openai", "litellm", or "http"
    """
    # For now, default to python adapter (safest for BYOA)
    # Future: Could analyze imports or base classes to detect OpenAI/LangGraph/etc
    return "python"
