# Stateful adapter sessions

Atlas now opens a single adapter session for each runtime task (plan → execute
→ synthesize). Adapters that expose the session interface can reuse expensive
resources and emit richer telemetry for continual learning.

## Authoring a stateful adapter

1. Implement `open_session(SessionContext)` and return an object exposing
   `session_id`, `metadata`, `step(...)`, and `close(...)`. The base
   `StatelessSession` proxy keeps existing adapters working without any
   changes.
2. Inherit from `StatefulAgentAdapterMixin` or set `supports_sessions = True` to
   let the runtime know sessions are available.
3. Propagate `adapter_session_id` inside `metadata` when calling external
   services so telemetry can be stitched back together later.

```python
from atlas.connectors.registry import AgentAdapter, SessionContext

class MyAdapter(StatefulAgentAdapterMixin, AgentAdapter):
    async def open_session(self, context: SessionContext):
        return MySession(context)
```

The Python adapter automatically detects classes with `on_open`, `on_step`, and
`on_close` hooks. See `examples/stateful_adapter_sql/` for a working SQLite
example.

## Using sessions from the runtime

The Student persona manages sessions via `async with student.session_scope(...)`
so adapters see a single logical conversation. Standalone adapters can use the
same pattern:

```python
async with adapter.session(SessionContext(task_id="demo", execution_mode="stepwise")) as session:
    first = await session.step("plan")
    second = await session.step("execute", metadata={"adapter_session_id": session.session_id})
```

Every call returns usage and tool events which the runtime aggregates into
`context.metadata["adapter_session"]`. The JSONL exporter now includes:

- `adapter_session_id`
- `adapter_usage` (prompt/completion tokens, call count)
- `adapter_events` (tool invocations, environment actions)

These fields make it easy to join runtime traces with the GRPO / RIM continual
learning pipelines described in the Atlas foundation paper.

## Database migration

Existing deployments should add the following columns:

```sql
ALTER TABLE sessions
    ADD COLUMN IF NOT EXISTS adapter_session_id TEXT,
    ADD COLUMN IF NOT EXISTS adapter_usage JSONB,
    ADD COLUMN IF NOT EXISTS adapter_events JSONB;

ALTER TABLE step_results
    ADD COLUMN IF NOT EXISTS adapter_session_id TEXT,
    ADD COLUMN IF NOT EXISTS adapter_usage JSONB,
    ADD COLUMN IF NOT EXISTS adapter_events JSONB;
```

Restart the runtime after applying the migration so new telemetry is persisted
automatically.
