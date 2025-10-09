# Telemetry & Runtime Data Flow

Atlas SDK keeps telemetry intentionally lightweight so existing agent builders can plug the runtime into their production
systems without standing up new tracing infrastructure. This note summarises the current pipeline, the data contract with
the Atlas core repository, and how to export traces for offline training.

---

## Runtime Event Stream

- Every `atlas.core.run` invocation initialises an `ExecutionContext`.  
- Components (student, teacher, evaluator, orchestration helpers) push `IntermediateStepPayload` objects through the
  context’s `IntermediateStepManager`, creating a lightweight in-process event stream.  
- The console renderer and Postgres persistence layer are the only subscribers today; there is no LangChain callback or
  OpenTelemetry dependency by default.

This design keeps local development fast—no background services or network calls—while still capturing the complete
workflow history when persistence is enabled.

---

## Data Contract with Atlas Core

The persisted data must match the schema expected by the training stack in
[Arc-Computer/ATLAS](https://github.com/Arc-Computer/ATLAS) (`atlas_core/runtime/schema.py`). The SDK currently records:

- **Session envelope** – task string, final answer, plan JSON, and user-supplied metadata.
- **Step traces** – description, executor trace, structured executor output (JSON with `status`, `artifacts`, optional `notes`),
  validation verdicts, retry guidance, attempt counts, and the full reward breakdown (`score`, per-judge details, tier samples).
  Reasoning blocks captured from the underlying LLM are attached under `metadata`.
- **Trajectory events** – serialized `IntermediateStepPayload` items describing workflow/tool/LLM boundaries for replay
  or advanced analytics.

Running `arc-atlas --database-url ... --output traces.jsonl` (or `python -m atlas.cli.export ...`) materialises the JSONL format consumed by
`trainers/runtime_dataset.py` in the core repo. Loading the file via `load_runtime_traces("traces.jsonl")` returns
`AtlasSessionTrace` objects without additional adapters.

---

## Developer Workflow

1. Configure PostgreSQL persistence in your Atlas config (`storage.database_url`).
2. Execute workloads with `atlas.core.run(...)`; telemetry remains in-process unless persistence is enabled.
3. Invoke `arc-atlas` (or `python -m atlas.cli.export`) to produce training-ready JSONL.
4. Feed the export into Atlas core utilities (`load_runtime_traces`, `flatten_traces_for_training`, etc.).

If you need the raw text or structured artifacts from a step, parse the `output` string with `json.loads(...)` and inspect the `status` / `artifacts`
fields before feeding the data into downstream pipelines.

There are no mandatory external tracing systems—developers can keep using their existing agent infrastructure and opt
into richer telemetry later if needed.

---

## Extensibility Guidelines

If you need deeper observability (e.g., LangChain callbacks or OpenTelemetry exporters):

- Prefer building on top of `IntermediateStepManager` to avoid schema drift.
- Preserve the `AtlasSessionTrace`/`AtlasStepTrace` fields so exported datasets remain compatible with the core repo.
- Document any additional metadata you surface so consumers can feature-gate on it.

By default, the SDK prioritises simplicity and data fidelity over heavy telemetry instrumentation. This keeps the
onboarding path short for teams that already operate agents in production while ensuring the exported traces slot
directly into the Atlas training stack.
