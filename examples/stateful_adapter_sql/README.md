# Stateful Python adapter example

This example shows how to build a stateful Python adapter that keeps a database
connection alive across multiple steps. The `SQLiteAgent` class exposes
`on_open`, `on_step`, and `on_close` hooks – when it is loaded through the
`PythonAdapter` the hooks are invoked for each `SessionContext` so the agent can
reuse connections and emit structured telemetry.

```python
from atlas.connectors.python import PythonAdapter
from atlas.config.models import AdapterType, PythonAdapterConfig
from atlas.connectors.registry import SessionContext

config = PythonAdapterConfig(
    type=AdapterType.PYTHON,
    name="sqlite-agent",
    system_prompt="",
    import_path="examples.stateful_adapter_sql.adapter",
    attribute="SQLiteAgent",
)
adapter = PythonAdapter(config)

session = await adapter.open_session(SessionContext(task_id="demo", execution_mode="stepwise"))
rows = await session.step("SELECT title FROM documents ORDER BY id")
await session.close()
```

See `tests/unit/test_adapter_sessions.py` for an end-to-end usage example that
verifies the session metadata captured by the runtime.
