"""LLM provider inference and configuration utilities.

This module provides functions for normalizing LLM provider names, inferring providers
from model names, and selecting optimal LLM configurations from metadata.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any


# Default environment variables for API keys by provider
LLM_PROVIDER_DEFAULT_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "azure-openai": "AZURE_OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "bedrock": "AWS_ACCESS_KEY_ID",
    "fireworks": "FIREWORKS_API_KEY",
    "cohere": "COHERE_API_KEY",
    "ai21": "AI21_API_KEY",
    "xai": "XAI_API_KEY",
}


def normalise_provider_name(provider: object) -> str | None:
    """Normalize LLM provider name to canonical lowercase form.

    Args:
        provider: Raw provider name (any type, but str expected)

    Returns:
        Normalized provider name, or None if invalid

    Examples:
        >>> normalise_provider_name("OpenAI")
        'openai'
        >>> normalise_provider_name("AzureOpenAI")
        'azure-openai'
        >>> normalise_provider_name("Google-GenerativeAI")
        'google'
    """
    if not isinstance(provider, str):
        return None
    normalised = provider.strip().lower().replace("_", "-")
    if not normalised:
        return None
    if normalised == "azureopenai":
        normalised = "azure-openai"
    if normalised in {"google-generativeai", "vertex-ai", "vertex"}:
        normalised = "google"
    return normalised


def infer_provider_from_model(model: object) -> str | None:
    """Infer LLM provider from model name using heuristics.

    Args:
        model: Model name (any type, but str expected)

    Returns:
        Inferred provider name, or None if unable to infer

    Examples:
        >>> infer_provider_from_model("gpt-4")
        'openai'
        >>> infer_provider_from_model("claude-3-5-sonnet")
        'anthropic'
        >>> infer_provider_from_model("gemini-pro")
        'gemini'
    """
    if not isinstance(model, str):
        return None
    value = model.strip().lower()
    if not value:
        return None
    if "claude" in value or "anthropic" in value or value.startswith("sonnet"):
        return "anthropic"
    if value.startswith("gpt") or "openai" in value or value.startswith("o4") or value.startswith("chatgpt"):
        return "openai"
    if value.startswith("gemini") or "google" in value:
        return "gemini"
    if "groq" in value:
        return "groq"
    if "mistral" in value:
        return "mistral"
    if "cohere" in value:
        return "cohere"
    if "grok" in value or "xai" in value:
        return "xai"
    if "bedrock" in value or value.startswith("anthropic:"):
        return "bedrock"
    return None


def _normalise_llm_candidate_entry(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize a single LLM configuration entry.

    Extracts and normalizes provider, model, API key env var, temperature, max_tokens, etc.
    """
    provider = normalise_provider_name(raw.get("provider") or raw.get("api_type"))
    model = raw.get("model") or raw.get("model_name")
    if isinstance(model, (list, tuple)):
        model = next((item for item in model if isinstance(item, str) and item.strip()), None)
    if not provider and isinstance(model, str):
        provider = infer_provider_from_model(model)
    candidate: dict[str, Any] = {}
    if provider:
        candidate["provider"] = provider
    if isinstance(model, str) and model.strip():
        candidate["model"] = model.strip()
    api_key_env = raw.get("api_key_env") or raw.get("api_key")
    if isinstance(api_key_env, str) and api_key_env.strip():
        candidate["api_key_env"] = api_key_env.strip()
    temperature = raw.get("temperature")
    if isinstance(temperature, (int, float)):
        candidate["temperature"] = float(temperature)
    max_tokens = raw.get("max_output_tokens")
    if not isinstance(max_tokens, (int, float)):
        max_tokens = raw.get("max_tokens")
    if isinstance(max_tokens, (int, float)):
        candidate["max_output_tokens"] = int(max_tokens)
    timeout = raw.get("timeout_seconds")
    if isinstance(timeout, (int, float)):
        candidate["timeout_seconds"] = float(timeout)
    source = raw.get("source")
    if isinstance(source, str) and source.strip():
        candidate["source"] = source
    return candidate or None


def _dedupe_candidate_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate LLM candidate entries based on provider/model pair."""
    seen: set[tuple[str | None, str | None]] = set()
    unique: list[dict[str, Any]] = []
    for entry in entries:
        provider = entry.get("provider")
        model = entry.get("model")
        key = (
            provider if isinstance(provider, str) else None,
            model if isinstance(model, str) else None,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    return unique


def collect_llm_candidate_entries(agent_runtime_meta: dict[str, object] | None) -> list[dict[str, Any]]:
    """Collect and normalize LLM configuration candidates from runtime metadata.

    Searches multiple metadata sources (llm_overrides, config_literals, config_data,
    llm_candidates, factory_kwargs) and returns deduplicated normalized entries.

    Args:
        agent_runtime_meta: Runtime metadata dictionary from discovery

    Returns:
        List of normalized LLM configuration dictionaries with source annotations
    """
    if not isinstance(agent_runtime_meta, dict):
        return []
    collected: list[dict[str, Any]] = []

    def _add_candidate(entry: dict[str, Any], source: str) -> None:
        payload = dict(entry)
        if source and "source" not in payload:
            payload["source"] = source
        normalised = _normalise_llm_candidate_entry(payload)
        if normalised:
            if source and "source" not in normalised:
                normalised["source"] = source
            collected.append(normalised)

    overrides = agent_runtime_meta.get("llm_overrides")
    if isinstance(overrides, dict) and overrides:
        _add_candidate(overrides, "factory_kwargs")

    config_literals = agent_runtime_meta.get("config_literals")
    if isinstance(config_literals, list):
        for literal in config_literals:
            if isinstance(literal, dict):
                _add_candidate(literal, "config_literals")
            elif isinstance(literal, list):
                for item in literal:
                    if isinstance(item, dict):
                        _add_candidate(item, "config_literals")

    config_data = agent_runtime_meta.get("config_data")
    if isinstance(config_data, list):
        for entry in config_data:
            if not isinstance(entry, dict):
                continue
            content = entry.get("content")
            if isinstance(content, dict):
                raw_path = entry.get("path")
                if isinstance(raw_path, Path):
                    source = str(raw_path)
                elif isinstance(raw_path, str):
                    source = raw_path
                else:
                    source = "config_file"
                payload = dict(content)
                payload.setdefault("source", source)
                _add_candidate(payload, source)

    llm_candidates_meta = agent_runtime_meta.get("llm_candidates")
    if isinstance(llm_candidates_meta, list):
        for entry in llm_candidates_meta:
            if isinstance(entry, dict):
                _add_candidate(entry, entry.get("source") or "metadata")

    factory_kwargs = agent_runtime_meta.get("factory_kwargs")
    if isinstance(factory_kwargs, dict):
        maybe_model = factory_kwargs.get("model")
        if isinstance(maybe_model, str):
            _add_candidate({"model": maybe_model}, "factory_kwargs")
        config_list = factory_kwargs.get("config_list")
        if isinstance(config_list, list):
            for item in config_list:
                if isinstance(item, dict):
                    _add_candidate(item, "config_list")

    return _dedupe_candidate_entries(collected)


def select_llm_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Select the best LLM configuration from a list of candidates.

    Scoring criteria (highest to lowest priority):
    1. Richness: presence of provider, model, api_key_env
    2. Completeness: both provider AND model present
    3. Nuance: presence of temperature, max_output_tokens

    Args:
        candidates: List of normalized LLM configuration dictionaries

    Returns:
        Best candidate configuration, or None if no candidates
    """
    if not candidates:
        return None

    def _score(entry: dict[str, Any]) -> tuple[int, int, float]:
        provider_present = int(bool(entry.get("provider")))
        model_present = int(bool(entry.get("model")))
        richness = provider_present * 2 + model_present * 3 + int(bool(entry.get("api_key_env")))
        nuance = 0.0
        if entry.get("temperature") is not None:
            nuance += 0.1
        if entry.get("max_output_tokens") is not None:
            nuance += 0.1
        return (richness, provider_present * model_present, nuance)

    candidates_sorted = sorted(
        candidates,
        key=_score,
        reverse=True,
    )
    return candidates_sorted[0]


def merge_llm_block(existing: dict[str, Any] | None, candidate: dict[str, Any]) -> dict[str, Any]:
    """Merge LLM configuration candidate into existing configuration block.

    Updates provider, model, api_key_env, temperature, max_output_tokens, timeout_seconds.
    Automatically sets default api_key_env if provider changes.

    Args:
        existing: Existing LLM configuration block (or None)
        candidate: New configuration to merge in

    Returns:
        Merged configuration dictionary
    """
    block = copy.deepcopy(existing) if existing else {}
    previous_provider = block.get("provider") if isinstance(block.get("provider"), str) else None
    candidate_api_env = candidate.get("api_key_env") if isinstance(candidate.get("api_key_env"), str) else None
    for key in ("provider", "model", "api_key_env", "temperature", "max_output_tokens", "timeout_seconds"):
        if key in candidate and candidate[key] is not None:
            block[key] = candidate[key]
    provider_value = block.get("provider")
    if isinstance(provider_value, str):
        normalised = normalise_provider_name(provider_value)
        if normalised:
            block["provider"] = normalised
            if candidate_api_env:
                block["api_key_env"] = candidate_api_env
            api_env = block.get("api_key_env")
            provider_changed = (previous_provider or "").lower() != normalised
            if (provider_changed or not api_env) and not candidate_api_env:
                default_env = LLM_PROVIDER_DEFAULT_ENV.get(normalised)
                if default_env:
                    block["api_key_env"] = default_env
        else:
            block.pop("provider", None)
    return block
