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

## Session-aware adapters

Adapters can opt-in to stateful behaviour by implementing the session protocol:

- Override `open_session(SessionContext) -> AgentSession` and set
  `supports_sessions = True` (or inherit from
  `StatefulAgentAdapterMixin`).
- The returned `AgentSession` must expose a `session_id`, `metadata`,
  `step(...)`, and `close(...)` method.
- Stateless adapters remain compatible via the built-in `StatelessSession`
  proxy, so existing integrations do not need to change.
- `atlas.connectors.python.SessionResourcePool` helps share long-lived handles
  (e.g. DB connections) across sessions.

`SessionContext` captures the task identifier, execution mode, available tools,
and arbitrary user context. When adapters expose sessions the runtime opens a
single session per Atlas task and guarantees `close()` is called even if
planning or execution fails.

### Python adapter hooks

The Python adapter automatically detects classes that implement
`on_open`, `on_step`, and `on_close` hooks. The hooks run exactly once per
session and can return additional metadata:

```python
class MyAgent:
    async def on_open(self, context: SessionContext) -> dict:
        self.cache = {...}
        return {"ready": True}

    async def on_step(self, prompt: str, metadata: dict | None = None):
        return {"content": self.cache[prompt]}

    async def on_close(self, reason: str | None = None):
        self.cache.clear()
```

See `examples/stateful_adapter_sql` for a complete adapter that shares a SQLite
connection across multiple steps.
