import types
from pathlib import Path

from atlas.cli.env import SelectedTargets, TargetSpec, _compose_full_config_payload
from atlas.config.models import AtlasConfig
from atlas.sdk.discovery_worker import _detect_llm_metadata


def test_detect_llm_metadata_from_langchain_like_stack() -> None:
    FakeChat = types.new_class("ChatOpenAI", (), {})
    FakeChat.__module__ = "langchain_openai.chat_models.base"

    instance = FakeChat()
    setattr(instance, "model_name", "gpt-4.1-mini")

    metadata = _detect_llm_metadata(instance)
    assert metadata["provider"] == "openai"
    assert metadata["model"] == "gpt-4.1-mini"
    assert metadata["source"].endswith("ChatOpenAI")


def test_compose_full_config_payload_merges_discovery_targets() -> None:
    template_payload = {
        "agent": {
            "type": "litellm",
            "name": "example-openai-agent",
            "system_prompt": "You are a student.",
            "tools": [],
            "llm": {
                "provider": "openai",
                "model": "gpt-4.0-mini",
                "api_key_env": "OPENAI_API_KEY",
            },
        },
        "teacher": {
            "llm": {
                "provider": "openai",
                "model": "gpt-4.0-mini",
                "api_key_env": "OPENAI_API_KEY",
            }
        },
        "student": {
            "max_plan_tokens": 1024,
            "max_step_tokens": 1024,
            "max_synthesis_tokens": 1024,
        },
        "rim": {
            "small_model": {
                "provider": "gemini",
                "model": "gemini/gemini-1.5-flash",
                "api_key_env": "GEMINI_API_KEY",
            },
            "large_model": {
                "provider": "gemini",
                "model": "gemini/gemini-1.5-flash",
                "api_key_env": "GEMINI_API_KEY",
            },
        },
        "storage": None,
    }
    targets = SelectedTargets(
        environment=TargetSpec(factory=("tests.fixtures.langgraph_adapter", "create_environment")),
        agent=TargetSpec(factory=("tests.fixtures.langgraph_adapter", "create_langgraph_agent")),
    )

    payload, info = _compose_full_config_payload(
        template_payload,
        targets,
        Path("/projects/demo"),
        {"provider": "anthropic", "model": "claude-3-sonnet"},
    )

    assert payload is not None
    assert payload["agent"]["type"] == "python"
    assert "response_format" not in payload["agent"]
    assert payload["agent"]["name"] == "example-openai-agent"
    assert payload["agent"]["tools"] == []
    assert payload["agent"]["import_path"] == "tests.fixtures.langgraph_adapter"
    assert payload["agent"]["attribute"] == "create_langgraph_agent"
    assert payload["agent"]["allow_generator"] is False
    assert payload["agent"]["llm"]["provider"] == "anthropic"
    assert payload["agent"]["llm"]["model"] == "claude-3-sonnet"
    assert payload["teacher"]["llm"]["provider"] == "anthropic"
    assert payload["teacher"]["llm"]["model"] == "claude-3-sonnet"
    assert isinstance(payload.get("learning"), dict)
    assert isinstance(payload.get("runtime_safety"), dict)

    metadata_block = payload.get("metadata", {}).get("discovery", {})
    assert metadata_block["agent_factory"]["module"] == "tests.fixtures.langgraph_adapter"
    assert metadata_block["environment_factory"]["module"] == "tests.fixtures.langgraph_adapter"
    assert info["llm_provider"] == "anthropic"
    assert info["llm_model"] == "claude-3-sonnet"


def test_full_config_payload_validates_against_model() -> None:
    template_payload = {
        "agent": {
            "type": "litellm",
            "name": "example-openai-agent",
            "system_prompt": "You are a student.",
            "tools": [],
            "llm": {
                "provider": "openai",
                "model": "gpt-4.0-mini",
                "api_key_env": "OPENAI_API_KEY",
            },
        },
        "teacher": {
            "llm": {
                "provider": "openai",
                "model": "gpt-4.0-mini",
                "api_key_env": "OPENAI_API_KEY",
            }
        },
        "student": {
            "max_plan_tokens": 1024,
            "max_step_tokens": 1024,
            "max_synthesis_tokens": 1024,
        },
        "rim": {
            "small_model": {
                "provider": "gemini",
                "model": "gemini/gemini-1.5-flash",
                "api_key_env": "GEMINI_API_KEY",
            },
            "large_model": {
                "provider": "gemini",
                "model": "gemini/gemini-1.5-flash",
                "api_key_env": "GEMINI_API_KEY",
            },
        },
        "storage": None,
    }
    targets = SelectedTargets(
        environment=TargetSpec(factory=("tests.fixtures.langgraph_adapter", "create_environment")),
        agent=TargetSpec(factory=("tests.fixtures.langgraph_adapter", "create_langgraph_agent")),
    )

    payload, _ = _compose_full_config_payload(
        template_payload,
        targets,
        Path("/projects/demo"),
        {"provider": "openai", "model": "gpt-4.1-mini"},
    )

    assert payload is not None
    config = AtlasConfig.model_validate(payload)
    assert config.agent.type.value == "python"
