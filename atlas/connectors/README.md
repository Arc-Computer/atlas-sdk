# atlas.connectors

`atlas.connectors` contains the Bring-Your-Own-Agent adapter registry and default implementations for HTTP, OpenAI, Python, and LangChain bridges.

## Adapter lifecycle

1. **Register** a builder with `atlas.connectors.register_adapter(AdapterType, builder)`.
2. **Instantiate** via `atlas.connectors.create_adapter(config)` or `create_from_atlas_config`.
3. **Execute** calls through the returned `AgentAdapter`, which exposes async (`ainvoke`) and sync (`execute`) entry points.

## Built-in adapters

- `http.py` – generic REST bridge for external agents.
- `openai.py` – thin wrapper around `litellm` for OpenAI-compatible endpoints.
- `python.py` – in-process Python callable adapter.
- `langchain_bridge.py` – LangChain LLM/tools integration for student personas.

Each module handles optional dependencies gracefully; consumers can register additional adapters without modifying Atlas core.

Related packages:

- `atlas.personas` – consumes adapters for Student/Teacher personas.
- `atlas.runtime.agent_loop` – executes adapter results within LangGraph.
