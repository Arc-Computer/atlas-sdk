# Stateful Adapter Handshake & Telemetry Contract

Partners that already run multi-step agents were forced to let Atlas drive the inner loop. Issue #65 introduces a capability handshake that lets adapters declare whether they retain control, while Atlas continues to run the outer loop (triage, teacher review, reward capture, export guardrails). This note documents the contract and the reference implementation that shipped with the runtime.

## Capability Handshake

Every adapter now receives an `aopen_session` call when a session starts:

```python
async def aopen_session(
    self,
    *,
    task: str,
    metadata: dict[str, Any] | None = None,
    emit_event: AdapterEventEmitter | None = None,
) -> AdapterCapabilities:
    ...
```

Return an `AdapterCapabilities` payload describing the inner-loop behaviour:

```json
{
  "control_loop": "self",
  "supports_stepwise": false,
  "telemetry_stream": true
}
```

- `control_loop`: `"atlas"` (default) keeps the existing prompt-driven flow. `"self"` tells Atlas to call the adapter once per outer-loop phase and skip intermediate retries.
- `supports_stepwise`: set to `true` only if the adapter can participate in Atlas’ step-by-step lanes (`coach` / `escalate`). When `false`, Atlas automatically routes self-managed adapters to the single-shot modes (`auto` or `paired`).
- `telemetry_stream`: advertise whether the adapter will use the supplied event emitter.

The handshake result is stored in `ExecutionContext.metadata["adapter_capabilities"]` and exported with the session metadata. Integrators can pin the behaviour in configuration with:

```yaml
agent:
  behavior: self  # or atlas
```

The runtime honours the override while still recording the true handshake payload for audit.

## Outer-Loop Hooks

When `control_loop="self"`, the `Student` façade stops rewriting prompts. Instead, Atlas calls the adapter once per outer-loop phase:

```python
await adapter.aplan(task, metadata=...)
await adapter.aexecute(task, plan, step, metadata=...)
await adapter.asynthesize(task, plan, step_results, metadata=...)
```

- `aplan` should return a `Plan` (or dict) describing the work the adapter will perform. The runtime enforces single-shot execution unless `supports_stepwise=True`.
- `aexecute` runs the entire inner loop and returns a mapping compatible with `StudentStepResult` (`trace`, `output`, `metadata`, optional `deliverable`, etc.).
- `asynthesize` produces the final answer given the plan and step summaries.

Atlas still:
- Triages the task and records metadata.
- Runs teacher review / validation when in `paired` mode.
- Captures reward statistics and drift signals.
- Streams intermediate steps (triage, routing, teacher verdicts) into Postgres.

## Adapter Telemetry Events

Stateful adapters receive an `emit_event` callback during the handshake. Emitters are awaitable callables that accept dictionaries or `AdapterTelemetryEvent` objects:

```python
await emit_event({
    "event": "env_action",           # env_action | tool_response | progress | error
    "payload": {"sql": "..."},
    "reason": "Deriving answer from demo dataset.",
    "step": 1
})
```

Atlas wraps each event into an `IntermediateStep` with type `ADAPTER_EVENT`. They flow through:
- `ExecutionContext.event_stream` (live subscribers).
- Console telemetry (`ADAPTER ...` lines).
- Postgres `trajectory_events`, so exports and replay tooling see adapter-originated traces.

The JSON envelope stored downstream contains `event`, `payload`, `reason`, `step`, `timestamp`, and optional `metadata`.

## Reference Example

`examples/adapters.py` implements `StatefulSQLiteAdapter`, a self-managed agent that queries a small in-memory SQLite dataset, emits telemetry, and returns a final answer without Atlas prompt rewrites. The matching configuration lives at `configs/examples/stateful_sqlite.yaml`:

```yaml
agent:
  type: python
  behavior: self
  import_path: examples.adapters
  attribute: StatefulSQLiteAdapter
```

Run it with a short script:

```python
from atlas import core

result = core.run(
    task="List the active Atlas SDK projects and their velocity.",
    config_path="configs/examples/stateful_sqlite.yaml",
)

print(result.final_answer)
```

You’ll see adapter events streamed in the console while the outer loop continues to triage, route, and capture reward stats.

## Operational Notes

- Handshake failures surface as `AdapterError` to make debugging explicit.
- If a self-managed adapter incorrectly reports `supports_stepwise=False` but later tries to emit step-level retries, Atlas still honours the single-shot contract.
- Telemetry emitters must be awaited; the runtime warns and drops malformed events instead of failing the session.

The design keeps stateless adapters untouched, while enabling stateful partners to bring their own orchestration without losing Atlas governance or learning signals.
