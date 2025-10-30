import pytest

from atlas.cli import env


def _load_helper_namespace():
    namespace: dict[str, object] = {}
    exec(env.AGENT_RUNTIME_HELPER, namespace)
    return namespace


def test_atlas_execute_agent_prefers_human_message():
    pytest.importorskip("langchain_core.messages")
    namespace = _load_helper_namespace()
    HumanMessage = namespace["HumanMessage"]
    assert HumanMessage is not None

    class GuardedAgent:
        def __init__(self):
            self.payloads = []

        def invoke(self, payload, **kwargs):
            self.payloads.append(payload)
            messages = payload.get("messages") if isinstance(payload, dict) else None
            if (
                isinstance(messages, list)
                and len(messages) == 1
                and isinstance(messages[0], HumanMessage)
            ):
                return "ok"
            raise TypeError("unsupported payload")

    result = namespace["_atlas_execute_agent"](GuardedAgent(), "ping")
    assert result == "ok"


def test_atlas_execute_agent_handles_missing_human_message():
    namespace = _load_helper_namespace()
    namespace["HumanMessage"] = None

    class DictAgent:
        def __init__(self):
            self.payloads = []

        def invoke(self, payload, **kwargs):
            self.payloads.append(payload)
            if payload == {"messages": [{"role": "user", "content": "ping"}]}:
                return "dict"
            raise TypeError("unsupported payload")

    result = namespace["_atlas_execute_agent"](DictAgent(), "ping")
    assert result == "dict"
