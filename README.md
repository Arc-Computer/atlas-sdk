# Atlas SDK

Atlas is infrastructure for continual learning in LLM agents. It wraps any agent (OpenAI, Claude, Gemini, local models, or your own stack) with a closed-loop Student → Teacher → Reward system so the agent improves from every interaction. The runtime SDK handles planning, orchestration, evaluation, telemetry, and exportable traces; the same data feeds Atlas core’s online and offline optimisation workflows.

[![PyPI version](https://img.shields.io/pypi/v/arc-atlas.svg)](https://pypi.org/project/arc-atlas/)
[![PyPI downloads](https://img.shields.io/pypi/dm/arc-atlas.svg)](https://pypi.org/project/arc-atlas/)

```bash
pip install arc-atlas
```

---

## 📚 Full Documentation

The README hits the highlights. For the complete guide—including configuration tables, orchestration deep dives, and training recipes—visit [docs.arc.computer](https://docs.arc.computer).

---

## Key Features

- **Bring-Your-Own-Agent (BYOA) Adapters** – Drop in HTTP, Python, or OpenAI agents without rewriting core logic.
- **Teacher / Student Loop** – Plans and executes tasks sequentially with review, validation, and retry guidance.
- **Reward System (RIM)** – Runs configurable judges (process, helpfulness, custom) to score every step.
- **Trajectory Capture** – Emits intermediate steps that can be streamed, logged, or audited later.
- **PostgreSQL Persistence** – Ships with an async persistence layer and schema for sessions, attempts, guidance, and events.

---

## Quick Start

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

Run an example configuration:

```python
from atlas import core

result = core.run(
    task="Summarise the latest financial news",
    config_path="configs/examples/openai_agent.yaml",
)

print(result.final_answer)
```

Atlas returns an `atlas.types.Result` containing the final answer, the reviewed plan, and per-step evaluations.

---

## Run with Docker

The repo ships with a ready-to-go Compose stack under `docker/`:

```bash
# 1. Ensure your project .env includes the required keys (Compose reads it automatically):
#    OPENAI_API_KEY=sk-...
#    GOOGLE_API_KEY=...
# 2. Build the SDK image and start Postgres + the demo agent
docker compose -f docker/docker-compose.yaml up --build
```

- `postgres` starts a local PostgreSQL instance with a persisted volume (`atlas_pg_data`).
- `atlas` builds the SDK, installs it in editable mode, and runs the ARC demo by default.
- Pass a custom command to run other configs:  
  `docker compose -f docker/docker-compose.yaml run --rm atlas python -m atlas.cli.main --help`

The container mounts your repo at `/workspace`, so you can edit code locally and rerun without rebuilding. The default entrypoint is `docker/entrypoint.sh`; override it by supplying arguments after the service name (they replace the demo command).

---

## Using `pip install arc-atlas`

When you install the SDK from PyPI you still need a PostgreSQL URL if you want persistence. The easiest path is to reuse the bundled Compose file:

```bash
pip install arc-atlas
docker compose -f docker/docker-compose.yaml up -d postgres

# Either export these for the current shell or ensure they're present in .env
export STORAGE__DATABASE_URL=postgresql://atlas:atlas@localhost:5433/atlas_arc_demo
export OPENAI_API_KEY=sk-...
# Optional Process/Helpfulness judges
export GOOGLE_API_KEY=...

python -m atlas.core.run --config path/to/config.yaml --task "Summarise the Atlas SDK"
```

- `docker-compose` exposes Postgres on host port `5433`; keep the URL in sync if you change the mapping.
- You can point `storage.database_url` inside your YAML config or rely on the `STORAGE__DATABASE_URL` environment variable shown above.
- If storage is optional for your workflow, set `storage: null` in the config—runs will skip persistence but still execute end-to-end.

---

## Exporting Runtime Sessions

Atlas persists full execution traces whenever PostgreSQL storage is configured. Convert those sessions into training-ready
JSONL with the bundled exporter:

```bash
# 1. Run tasks that log to Postgres (configure storage.database_url in your AtlasConfig)
atlas.core.run(...)

# 2. Export the captured sessions to JSONL (use the unique CLI name to avoid PATH collisions)
arc-atlas --database-url postgresql://localhost:5432/atlas --output traces.jsonl --limit 25

# (Fall back to python -m if shell PATH ordering is unpredictable)
python -m atlas.cli.export --database-url postgresql://localhost:5432/atlas --output traces.jsonl --limit 25

# 3. Load the dataset inside the Atlas core repo
from trainers.runtime_dataset import load_runtime_traces
sessions = load_runtime_traces("traces.jsonl")
```

The CLI accepts repeatable filters such as `--session-id`, `--status`, and `--trajectory-event-limit`. Pass a standard
PostgreSQL URL (including credentials) via `--database-url`. The exporter prints friendly counts of the sessions and steps
written and emits newline-delimited JSON—one session per line.

Each session record follows the shared runtime schema consumed by the training stack:

- `task`, `final_answer`, `plan` – orchestration metadata for the run.
- `session_metadata` – persisted metadata plus status/timestamps.
- `steps` – executor traces with descriptions, reward breakdowns, validation results, retry guidance, structured executor outputs, and telemetry metadata.
- `trajectory_events` – optional array of intermediate telemetry events for richer replay and debugging.

Once exported you can feed the file directly into `load_runtime_traces` or flatten it for RL pipelines with helpers in
`trainers/runtime_dataset.py` from the core repository.

---

## Configuration Guide

Configuration files live in `configs/examples/`. Each YAML document is validated against `atlas.config.models.AtlasConfig`.

| Section | Purpose |
| ------- | ------- |
| `agent` | Adapter settings (endpoint, Python import path, OpenAI model) and tool schemas |
| `student` | Planner / executor / synthesizer prompts and token limits |
| `teacher` | LLM parameters for plan review, validation, and retry guidance |
| `orchestration` | Retry policy, per-step timeout, and trajectory emission flags |
| `rim` | Judge definitions, weights, aggregation strategy, thresholds |
| `storage` | Optional PostgreSQL connection info for persistence |

Atlas combines your adapter's `system_prompt` with opinionated templates from `atlas.prompts` to construct the built-in personas:

1. **Planner Student** – drafts a dependency-aware plan
2. **Executor Student** – runs each step and returns a trace
3. **Teacher** – reviews plans, validates execution, and issues retries/guidance

Override the defaults by providing explicit `student.prompts` and `teacher.prompts` blocks in your configuration. Each template can reference `{base_prompt}` to splice in the adapter prompt.

### Example: HTTP Adapter (excerpt)

```yaml
agent:
  type: http_api
  name: example-http-agent
  system_prompt: |
    You are an HTTP-based agent that can call external services.
  tools:
    - name: web_search
      description: Search the web for relevant documents.
      parameters:
        type: object
        properties:
          query:
            type: string
            description: Query string to search for.
        required: [query]
  transport:
    base_url: http://localhost:8080/agent
    timeout_seconds: 60
```

---

## Architecture

```
1. core.run()                 # load config, adapter, context
2. Student.create_plan()      # ATLAS-derived planning graph via BYOA bridge
3. Teacher.review_plan()      # validates dependencies and tools
4. Orchestrator.arun()        # sequential execution, retries, telemetry
5. Evaluator.ajudge()         # process/helpfulness judges aggregate scores
6. Database.log_*()           # optional persistence of plans, attempts, trajectory events
```

Trajectory events stream through `ExecutionContext.event_stream`, enabling live console streaming and durable storage via `atlas/runtime/storage/database.py` and `atlas/runtime/storage/schema.sql`.

**RIM Model Guidance**

- Tier-1 judges (process/helpfulness): Gemini 2.5 Flash or Grok-4 Fast provide fast, low-cost scores.
- Tier-2 arbiter: Gemini 2.5 Pro reconciles disagreements with high fidelity.
- Supplied examples show how to point `rim.judges[].llm` and `rim.arbiter` at different providers if desired.

---

## Terminal Telemetry

Atlas streams orchestration events directly to the terminal when `core.run` executes in an interactive shell. The default console renderer highlights the accepted plan, step attempts, tool invocations, reward scores, and the final synthesis without extra setup.

Example session:

```text
=== Atlas task started: Summarize the Atlas SDK (2025-02-12 10:15:03) ===
Plan ready with steps:
  1. gather dataset A
  2. synthesise findings
[step 1] attempt 1 started: gather dataset A
[tool] web_search call -> {"query": "Atlas SDK release"}
[tool] web_search result <- {"result": "..."}
[step 1] completed: gather dataset A
  reward score: 0.91
[step 2] retry 2 started: synthesise findings
  guidance: cite the repository README
=== Atlas task completed in 12.4s ===
Final answer:
  Atlas SDK ships a teacher-student loop...
- gather dataset A | attempts: 1 | score: 0.91
- synthesise findings | attempts: 2 | score: 0.88
RIM scores | max: 0.91 | avg: 0.89
```

Disable streaming with `core.run(..., stream_progress=False)` when piping output or running in CI. Pass `stream_progress=True` to force streaming even when stdout is not a TTY. The renderer also works with `core.arun` and runs alongside PostgreSQL persistence, so stored sessions retain full telemetry.

See `docs/examples/terminal_telemetry.md` for a step-by-step walkthrough.

For a deeper look at how these events map onto the Atlas training stack—and why the SDK keeps telemetry lightweight—see
`docs/telemetry_overview.md`.

---

## Exporting Runtime Sessions

When persistence is enabled, every run captures plans, telemetry, and reward data. Convert those sessions into JSONL with the `arc-atlas` CLI:

```bash
arc-atlas \
  --database-url postgresql://atlas:atlas@localhost:5432/atlas \
  --output traces.jsonl \
  --limit 25 \
  --trajectory-event-limit 500
```

Compatibility aliases `atlas.export` and `atlas-export` remain available, but they may collide with other tools named `atlas` if those appear earlier in your `PATH`. `arc-atlas` and `python -m atlas.cli.export` are collision-proof.

Key flags:

- `--session-id` (repeatable) restricts the export to explicit sessions.
- `--limit`/`--offset` and `--batch-size` page through large archives.
- `--trajectory-limit` controls how many intermediate events are embedded per session.

Each line in the output is an `AtlasSessionTrace` record:

```json
{
  "task": "Summarize the Atlas SDK",
  "final_answer": "The SDK wraps BYOA agents with a Student/Teacher loop...",
  "plan": {"steps": [...]},
  "steps": [
    {
      "step_id": 1,
      "description": "...",
      "tool": "summariser",
      "reward": {"score": 0.92, "judges": [...]},
      "validation": {"valid": true},
      "guidance": ["..."],
      "context": {"prior_results": {"1": "..."}},
      "artifacts": {"final_answer": "Paris"},
      "status": "ok",
      "output": "{\"status\":\"ok\",\"artifacts\":{\"final_answer\":\"Paris\"}}"
    }
  ],
  "session_metadata": {
    "session_id": 42,
    "status": "succeeded",
    "trajectory_events": [...]
  }
}
```

The structure aligns with `AtlasSessionTrace`, `AtlasStepTrace`, and `AtlasRewardBreakdown` used by `trainers/runtime_dataset.py`, so you can immediately consume the file inside the core repo:

1. Run `atlas.core.run(...)` with PostgreSQL persistence enabled.
2. Execute `arc-atlas --database-url ... --output traces.jsonl`.
3. Call `load_runtime_traces("traces.jsonl")` (from the core repo) to build training datasets.

Because each step output is now a JSON object, parse the string to access `status` and `artifacts` (e.g. `json.loads(step_output)["artifacts"]["final_answer"]`). Examples live in `docs/sdk/export-traces.mdx`.

---

## Testing

```bash
PYTHONPATH=. pytest tests --disable-warnings
```

The suite covers dependency parsing, prompt rewriting, student/teacher orchestration, RIM aggregation, adapter bridges, and database logging. Most tests rely on locally mocked adapters, so no external network calls occur.

---

## Requirements & Notes

- Python 3.10+ (project is developed and validated with 3.13).
- Development extras (`pip install -e .[dev]`) install pytest tooling for local validation; core telemetry streams rely solely on the standard library.
- Vendored NeMo components live under `atlas/roles/` and `atlas/utils/reactive/`; SPDX headers are retained and must remain intact.
- Aim for descriptive naming and concise docstrings so the intent is evident without extra commentary.

---

## Contributing

1. Fork and clone the repository.
2. Use the provided `pyproject.toml` extras to install development dependencies.
3. Review existing modules before coding and keep commits focused and incremental to match the current style.
4. Add or update unit tests alongside feature changes.

Pull requests should include updated documentation or examples when behaviour changes.

---

## License

Atlas SDK is released under the Apache 2.0 license. See `LICENSE` for full details. Vendored NeMo components retain their original licensing notices.

---

Need more depth or end-to-end walkthroughs? Everything in this README is covered—and expanded—at [docs.arc.computer](https://docs.arc.computer).
