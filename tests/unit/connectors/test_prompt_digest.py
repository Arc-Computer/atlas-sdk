import json

import pytest

from atlas.config.models import LLMParameters, LLMProvider, MetadataDigestConfig
from atlas.connectors.prompt_digest import build_prompt_digest, PromptDigestTooLargeError


def _llm_params(provider: LLMProvider = LLMProvider.ANTHROPIC) -> LLMParameters:
    return LLMParameters(provider=provider, model="test-model", api_key_env="TEST_KEY")


def test_build_prompt_digest_trims_large_sections():
    metadata = {
        "task": "Summarise incident" * 1000,
        "session_learning_audit": ["payload"] * 200,
        "session_reward_audit": [
            {"score": 1, "rationale": "rationale" * 200},
        ]
        * 50,
        "session_metadata": {"token_usage": {"prompt_tokens": 12345, "completion_tokens": 6789}},
    }
    digest_json = build_prompt_digest(metadata, _llm_params())
    digest = json.loads(digest_json)
    assert digest["digest_stats"]["size"] <= 20000
    assert "session_learning_audit" in digest["digest_stats"].get("omitted", [])
    audit_summary = digest["session"]["session_reward_audit_summary"]
    assert audit_summary["entries"] == 50
    assert "payload" not in audit_summary["sample"][0]
    section_sizes = digest["digest_stats"].get("sections") or {}
    assert isinstance(section_sizes, dict)
    assert sum(section_sizes.values()) <= digest["digest_stats"]["size"]
    assert digest["digest_stats"]["util"] == pytest.approx(
        digest["digest_stats"]["size"] / digest["digest_stats"]["budget"], rel=1e-6
    )


def test_build_prompt_digest_disabled_returns_original():
    metadata = {"foo": "bar"}
    digest_json = build_prompt_digest(metadata, _llm_params(), MetadataDigestConfig(enabled=False))
    assert json.loads(digest_json) == metadata


def test_build_prompt_digest_raises_when_budget_impossible():
    metadata = {"task": "x" * 5000}
    config = MetadataDigestConfig(provider_char_budgets={LLMProvider.ANTHROPIC: 1024}, max_section_chars=2000)
    with pytest.raises(PromptDigestTooLargeError):
        build_prompt_digest(metadata, _llm_params(), config)


def test_build_prompt_digest_warns_when_budget_near_limit(caplog):
    metadata = {
        "task": "a" * 400,
        "session_reward_audit": ["x" * 120] * 10,
        "session_metadata": {"source": "test"},
    }
    baseline = MetadataDigestConfig(char_budget=50000)
    digest = build_prompt_digest(metadata, _llm_params(), baseline)
    approx_size = len(digest)
    caplog.clear()
    warning_cfg = MetadataDigestConfig(char_budget=int(approx_size * 1.05))
    with caplog.at_level("WARNING"):
        build_prompt_digest(metadata, _llm_params(), warning_cfg)
    assert any("metadata digest consuming" in message for message in caplog.messages)


def test_build_prompt_digest_handles_stats_overhead_without_error():
    metadata = {
        "task": "x" * 5000,
        "plan": {"steps": [{"id": i, "description": "d" * 500} for i in range(10)]},
        "session_metadata": {"source": "s" * 1000, "execution_mode": "auto"},
        "steps": {"1": {"status": "ok", "attempts": []}},
    }
    config = MetadataDigestConfig(char_budget=4400, max_section_chars=1000, max_string_chars=500)
    digest_json = build_prompt_digest(metadata, _llm_params(), config)
    digest = json.loads(digest_json)
    assert digest["digest_stats"]["size"] <= config.char_budget
