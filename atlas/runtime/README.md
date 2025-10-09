# atlas.runtime

`atlas.runtime` bundles the orchestration, telemetry, storage, and schema layers that power Atlas sessions.

## Sub-packages

- `agent_loop/` – LangGraph agents (base agent, dual node, tool-calling loop, retries).
- `orchestration/` – Orchestrator, dependency graph, execution context, and intermediate step manager.
- `telemetry/` – Console streamer and LangChain callback that capture runtime events.
- `storage/` – PostgreSQL persistence (asyncpg pool + schema helpers).
- `models/` – Pydantic data models for intermediate steps and invocation nodes.
- `schema.py` – Serializable dataclasses (`AtlasSessionTrace`, `AtlasStepTrace`, `AtlasRewardBreakdown`).

## Extension guidance

- Subclass `agent_loop` components to alter LangGraph behaviour without touching persona logic.
- Use `telemetry` hooks to stream or persist additional diagnostics.
- Implement alternative persistence layers by following the `storage.Database` interface.
- Treat schema dataclasses as stable contracts consumed by the exporter and Atlas core training stack.

For high-level orchestration see `atlas.core`, which wires Student, Teacher, Evaluator, and Database together via this package.
