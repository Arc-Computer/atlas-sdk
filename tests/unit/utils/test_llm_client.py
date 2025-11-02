"""Tests for LLMClient offline mode functionality (Issue #110)."""

from __future__ import annotations

import warnings
from unittest.mock import patch

import pytest

from atlas.config.models import LLMParameters, LLMProvider
from atlas.utils.llm_client import LLMClient


def test_offline_mode_with_new_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that ATLAS_OFFLINE_MODE=1 works correctly."""
    monkeypatch.setenv("ATLAS_OFFLINE_MODE", "1")
    monkeypatch.delenv("ATLAS_FAKE_LLM", raising=False)

    params = LLMParameters(provider=LLMProvider.OPENAI, model="gpt-4o-mini", api_key_env="OPENAI_API_KEY")
    client = LLMClient(params)

    assert client._mock_mode is True


def test_offline_mode_with_legacy_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that ATLAS_FAKE_LLM=1 works with DeprecationWarning."""
    monkeypatch.delenv("ATLAS_OFFLINE_MODE", raising=False)
    monkeypatch.setenv("ATLAS_FAKE_LLM", "1")

    params = LLMParameters(provider=LLMProvider.OPENAI, model="gpt-4o-mini", api_key_env="OPENAI_API_KEY")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        client = LLMClient(params)

        assert client._mock_mode is True
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "ATLAS_FAKE_LLM is deprecated" in str(w[0].message)


def test_offline_mode_new_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that ATLAS_OFFLINE_MODE overrides ATLAS_FAKE_LLM when both are set."""
    monkeypatch.setenv("ATLAS_OFFLINE_MODE", "1")
    monkeypatch.setenv("ATLAS_FAKE_LLM", "1")

    params = LLMParameters(provider=LLMProvider.OPENAI, model="gpt-4o-mini", api_key_env="OPENAI_API_KEY")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        client = LLMClient(params)

        assert client._mock_mode is True
        # Should not warn when ATLAS_OFFLINE_MODE is set (even if ATLAS_FAKE_LLM is also set)
        deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 0


def test_offline_mode_mock_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify mock messages don't contain 'FAKE LLM' text."""
    monkeypatch.setenv("ATLAS_OFFLINE_MODE", "1")

    params = LLMParameters(provider=LLMProvider.OPENAI, model="gpt-4o-mini", api_key_env="OPENAI_API_KEY")
    client = LLMClient(params)

    # Test regular response
    response = client._mock_response([{"content": "test"}], None)
    assert "FAKE LLM" not in response.content
    assert "ATLAS_OFFLINE_MODE" in response.content

    # Test JSON response
    response_json = client._mock_response([{"content": "test"}], {"type": "json_object"})
    assert "FAKE LLM" not in response_json.content
    assert "ATLAS_OFFLINE_MODE" in response_json.content


def test_offline_mode_false_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that various false values don't enable offline mode."""
    for false_value in ["0", "", "false", "False"]:
        monkeypatch.setenv("ATLAS_OFFLINE_MODE", false_value)
        monkeypatch.delenv("ATLAS_FAKE_LLM", raising=False)

        params = LLMParameters(provider=LLMProvider.OPENAI, model="gpt-4o-mini", api_key_env="OPENAI_API_KEY")
        client = LLMClient(params)

        assert client._mock_mode is False, f"Expected False for ATLAS_OFFLINE_MODE={false_value}"


def test_offline_mode_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that offline mode is False when neither variable is set."""
    monkeypatch.delenv("ATLAS_OFFLINE_MODE", raising=False)
    monkeypatch.delenv("ATLAS_FAKE_LLM", raising=False)

    params = LLMParameters(provider=LLMProvider.OPENAI, model="gpt-4o-mini", api_key_env="OPENAI_API_KEY")
    client = LLMClient(params)

    assert client._mock_mode is False

