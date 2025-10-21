# Atlas Adapter Handshake Guide

This guide shows you how to plug an existing agent into the Atlas runtime while keeping your own control loop. By implementing a small handshake plus three async hooks you can let Atlas record telemetry, run governance (teacher, reward, review), and export datasets without prompt rewrites or retries in the middle of your run.

---

## Prerequisites

- Python 3.10+ with the `atlas` package installed in your environment.
- An adapter that implements the `AgentAdapter` protocol (see `atlas/connectors/registry.py`).
- Access to the Atlas config file that wraps your adapter (YAML).

If you are starting from scratch, copy the sample implementation in `examples/adapters.py` and the config in `configs/examples/adapters.yaml` to your project.

---

## 1. Advertise Capabilities During Session Handshake

When Atlas opens a session it calls `aopen_session`. Your adapter must return an `AdapterCapabilities` object (or dict) describing how you want Atlas to orchestrate the run.

```python
from atlas.connectors import AdapterCapabilities, AgentAdapter, AdapterEventEmitter

class MyAdapter(AgentAdapter):
    async def aopen_session(
        self,
        *,
        task: str,
        metadata: dict[str, str] | None = None,
        emit_event: AdapterEventEmitter | None = None,
    ) -> AdapterCapabilities:
        self._emit_event = emit_event
        return AdapterCapabilities(
            control_loop="self",      # you own planning/execution
            supports_stepwise=False, # Atlas should stay on auto/paired lanes
            telemetry_stream=True,   # you will emit adapter events
        )
```

Key fields:

| Field | Description |
|-------|-------------|
| `control_loop` | `"atlas"` (apply prompts + retries) or `"self"` (your adapter owns the inner loop). |
| `supports_stepwise` | `True` only if you can participate in Atlas’ step-by-step lanes (`coach`/`escalate`). |
| `telemetry_stream` | Set to `True` to opt into event streaming (see Section 3). |

The effective handshake payload (after any config override) is stored in `ExecutionContext.metadata["adapter_capabilities"]` and exported with the session record. The adapter’s original response is available at `ExecutionContext.metadata["adapter_capabilities_reported"]` for audit trails. Operators can override the behaviour in YAML by setting `agent.behavior: atlas|self`.

---

## 2. Implement Outer-Loop Hooks

Self-managed adapters must implement three async hooks. Atlas calls each hook once per session to cover plan, execute, and synthesis phases. Stateless adapters can continue using `ainvoke` only.

```python
from atlas.types import Plan, Step
from atlas.personas.student import StudentStepResult

class MyAdapter(AgentAdapter):
    ...
    async def aplan(self, task: str, metadata: dict | None = None) -> dict:
        # return dict that validates against atlas.types.Plan
        return {
            "steps": [{
                "id": 1,
                "description": "Run entire task in one shot.",
                "depends_on": [],
                "tool": None,
                "tool_params": None,
            }],
            "execution_mode": "single_shot",
        }

    async def aexecute(
        self,
        task: str,
        plan: dict,
        step: dict,
        metadata: dict | None = None,
    ) -> dict:
        # perform your inner loop and return StudentStepResult-compatible dict
        result_text = await self._run_agent(task)
        return {
            "trace": "...",              # string (for replay/debug)
            "output": result_text,       # string (final student output)
            "metadata": {"status": "ok", "reason": "..."},
            "deliverable": result_text,  # optional rich deliverable
        }

    async def asynthesize(
        self,
        task: str,
        plan: dict,
        step_results: list[dict],
        metadata: dict | None = None,
    ) -> str:
        return step_results[0].get("deliverable", "") if step_results else ""
```

Atlas still handles:

- Triage and routing (auto / paired / coach / escalate) based on your capabilities.
- Teacher validation and reward evaluation.
- Persistence to Postgres and export reviews.

---

## 3. Stream Adapter Telemetry (Optional but Recommended)

The `emit_event` callback provided during `aopen_session` lets you push telemetry into Atlas’ trajectory stream. Emit structured dictionaries or `AdapterTelemetryEvent` objects:

```python
await self._emit_event({
    "event": "env_action",            # env_action | tool_response | progress | error
    "payload": {"tool": "search", "query": q},
    "reason": "Kick off retrieval",
    "step": 1,
})
```

Atlas wraps the payload in an `ADAPTER_EVENT` intermediate step so it reaches:

- `ExecutionContext.event_stream` subscribers.
- Console telemetry (`ADAPTER …` lines).
- Postgres `trajectory_events` (and downstream exports).

Include enough detail for downstream analysis (what happened, why, and any context fields your replay tooling expects).

---

## 4. Configure Atlas to Load Your Adapter

Add the adapter to your Atlas configuration. Example:

```yaml
agent:
  type: python
  name: adapters-sqlite-agent
  import_path: my_project.adapters
  attribute: SQLiteAdapter
  behavior: self          # prefer adapter-managed mode
  tools: []

student:
  prompts:
    planner: "{base_prompt}"
    executor: "{base_prompt}"
    synthesizer: "{base_prompt}"
```

If you need to force Atlas-managed prompts (e.g., for comparison runs), set `behavior: atlas` without changing code.

---

## 5. Inspect Negotiated Capabilities from the CLI

Use the CLI helper to verify the handshake for any config:

```bash
atlas adapters describe --config configs/examples/adapters.yaml --task "demo"
```

The command prints the capabilities negotiated via `aopen_session`, letting you confirm routing before launching a full run.

---

## 6. End-to-End Example

The repo ships a working integration that you can adapt:

- Code: `examples/adapters.py` (`SQLiteAdapter`)
- Config: `configs/examples/adapters.yaml`
- Test: `tests/unit/test_adapters.py`

Try it with:

```python
from atlas import core

result = core.run(
    task="List the active Atlas SDK projects and their velocity.",
    config_path="configs/examples/adapters.yaml",
)
print(result.final_answer)
```

During the run you’ll see adapter telemetry on the console and the resulting session metadata will include your capability payload and emitted events.

---

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Adapter still runs step-by-step under Atlas control. | `control_loop` returned `"atlas"` or YAML override set to `atlas`. | Return `"self"` in `AdapterCapabilities` and remove the override. |
| Adapter events missing from exports. | Not emitting telemetry or `telemetry_stream=False`. | Ensure `telemetry_stream=True` and call `emit_event`. |
| Handshake raises `AdapterError`. | Capabilities payload fails validation. | Return an `AdapterCapabilities` instance or dict with the expected keys/values. |
| Teacher retry logic appears in your run. | `supports_stepwise=True` or lane override requires validation (paired mode). | Set `supports_stepwise=False` to stay in single-shot lanes or disable retries in config. |

If you run into edge cases, open an issue referencing the handshake contract so we can extend the schema safely.
