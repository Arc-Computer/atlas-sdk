import json
import pytest

pytest.importorskip("litellm")

from litellm.types.utils import Choices, ModelResponse, Usage

from atlas.connectors.openai import OpenAIAdapter
from atlas.config.models import LLMParameters, OpenAIAdapterConfig


def build_adapter():
    config = OpenAIAdapterConfig(
        name="test-openai",
        system_prompt="Base system",
        tools=[],
        llm=LLMParameters(model="gpt-test"),
    )
    return OpenAIAdapter(config)


def test_openai_adapter_builds_messages_from_metadata():
    adapter = build_adapter()
    metadata_messages = [
        {"type": "system", "content": "Meta system"},
        {"type": "human", "content": "Hello"},
        {
            "type": "ai",
            "content": "Thinking",
            "tool_calls": [
                {"name": "search", "arguments": {"query": "foo"}, "id": "call-1"},
                json.dumps({"name": "math", "arguments": {"expression": "1+1"}, "id": "call-2"}),
            ],
        },
        {"type": "tool", "content": {"result": "bar"}, "tool_call_id": "call-1"},
    ]
    messages = adapter._build_messages("Prompt", {"messages": metadata_messages})
    assert messages[0] == {"role": "system", "content": "Base system"}
    assert messages[1] == {"role": "system", "content": "Meta system"}
    assert messages[2] == {"role": "user", "content": "Hello"}
    assistant_message = messages[3]
    assert assistant_message["role"] == "assistant"
    assert len(assistant_message["tool_calls"]) == 2
    tool_message = messages[4]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call-1"
    assert "result" in tool_message["content"]
    assert messages[5] == {"role": "user", "content": "Prompt"}


def test_openai_adapter_trims_large_metadata_blob():
    adapter = build_adapter()
    oversized = "payload" * 50000
    metadata = {"session_learning_audit": [oversized]}
    messages = adapter._build_messages("Prompt", metadata)
    system_message = messages[1]
    assert system_message["role"] == "system"
    digest = json.loads(system_message["content"])
    stats = digest["digest_stats"]
    assert stats["size"] <= stats["budget"]
    assert len(system_message["content"]) == stats["size"]
    assert "payload" * 100 not in system_message["content"]


def test_openai_adapter_parses_usage_model_response():
    adapter = build_adapter()
    usage = Usage(prompt_tokens=12, completion_tokens=8, total_tokens=20)
    choice = Choices(
        finish_reason="stop",
        index=0,
        message={"content": "Answer", "role": "assistant"},
    )
    response = ModelResponse(id="resp-1", choices=[choice], usage=usage)
    parsed = adapter._parse_response(response)
    assert isinstance(parsed, str)
    assert parsed == "Answer"
    assert getattr(parsed, "tool_calls") in (None, [])
    assert getattr(parsed, "usage")["prompt_tokens"] == 12
    assert getattr(parsed, "usage")["completion_tokens"] == 8
    assert getattr(parsed, "usage")["total_tokens"] == 20


def test_openai_adapter_normalises_tool_calls():
    adapter = build_adapter()
    response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"name": "search", "arguments": json.dumps({"query": "atlas"}), "id": "call-1"},
                    ],
                }
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
    parsed = adapter._parse_response(response)
    assert isinstance(parsed, str)
    assert parsed == ""
    tool_calls = getattr(parsed, "tool_calls")
    assert tool_calls and tool_calls[0]["arguments"]["query"] == "atlas"
    usage = getattr(parsed, "usage")
    assert usage["total_tokens"] == 3
