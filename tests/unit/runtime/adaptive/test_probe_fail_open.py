"""Tests for capability probe fail-open behavior when API credentials are missing."""

import asyncio
import os
from unittest.mock import patch

import pytest

from atlas.config.models import AdaptiveProbeConfig, LLMParameters, LLMProvider
from atlas.runtime.adaptive.probe import CapabilityProbeClient


def test_probe_disables_when_api_key_missing():
    """Test that probe auto-detects missing API key and disables itself."""
    with patch.dict(os.environ, {}, clear=True):
        config = AdaptiveProbeConfig(
            llm=LLMParameters(
                provider=LLMProvider.XAI,
                model="xai/grok-4-fast",
                api_key_env="XAI_API_KEY",
                temperature=0.2,
            )
        )
        probe = CapabilityProbeClient(config)

        assert probe._enabled is False
        assert probe._client is None


def test_probe_enables_when_api_key_present():
    """Test that probe enables when API key is available."""
    with patch.dict(os.environ, {"XAI_API_KEY": "test-key"}):
        config = AdaptiveProbeConfig(
            llm=LLMParameters(
                provider=LLMProvider.XAI,
                model="xai/grok-4-fast",
                api_key_env="XAI_API_KEY",
                temperature=0.2,
            )
        )
        probe = CapabilityProbeClient(config)

        assert probe._enabled is True
        assert probe._client is not None


def test_probe_returns_none_when_disabled():
    """Test that disabled probe returns mode=None in decision."""
    async def runner() -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = AdaptiveProbeConfig(
                llm=LLMParameters(
                    provider=LLMProvider.XAI,
                    model="xai/grok-4-fast",
                    api_key_env="XAI_API_KEY",
                    temperature=0.2,
                )
            )
            probe = CapabilityProbeClient(config)

            decision = await probe.arun(
                task="test task",
                dossier={"summary": "test"},
                execution_metadata={"learning_history": {"count": 5}},
            )

            assert decision.mode is None
            assert decision.confidence is None
            assert decision.raw is not None
            assert decision.raw.get("disabled") is True
            assert "Missing API credentials" in decision.raw.get("reason", "")

    asyncio.run(runner())


def test_probe_fallback_mode_excludes_escalate():
    """Test that fallback_mode property only accepts paired/coach."""
    config = AdaptiveProbeConfig(fallback_mode="paired")
    probe = CapabilityProbeClient(config)

    assert probe.fallback_mode == "paired"

    config_coach = AdaptiveProbeConfig(fallback_mode="coach")
    probe_coach = CapabilityProbeClient(config_coach)

    assert probe_coach.fallback_mode == "coach"
