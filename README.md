# Atlas SDK
[![Atlas SDK hero](public/atlas-sdk.jpeg)](public/atlas-sdk.jpeg)

Atlas enables continual learning for agents to adapt, learn, and transfer knowledge in live environments.

[![PyPI version](https://img.shields.io/pypi/v/arc-atlas.svg)](https://pypi.org/project/arc-atlas/)
[![PyPI downloads](https://img.shields.io/pypi/dm/arc-atlas.svg)](https://pypi.org/project/arc-atlas/)

Atlas SDK is the continual-learning runtime that turns every task into a structured learning episode. It wraps any agent (OpenAI, Claude, Gemini, local models, or your own stack) with an adaptive dual-agent reasoning loop guided by reward signals, so agents stay fast on familiar work while escalating supervision on new or risky tasks. The SDK records rich telemetry, surfaces adaptive signals in real time, and exports production data for downstream training.

> **How it relates to [ATLAS](https://github.com/Arc-Computer/ATLAS)**  
> This repository delivers the runtime harness that powers continual learning in production. The `ATLAS` repo focuses on training models that ingest the structured traces produced here. Run the SDK to capture adaptive episodes; feed those traces into ATLAS to retrain and evaluate new policies.

---

With the split between SDK (runtime) and ATLAS (training) in mind, here's what our runtime gives you out of the box.

## Key Highlights (v0.1.3)

- **Adaptive Runtime** – Every request is triaged up front. We run a quick “can the agent handle this?” probe and pick the right lane: stay fully automated when confidence is high, ask the teacher to double-check the final answer, or run step-by-step with retries when risk is higher.
- **Persistent Learning Memory** – After each task, we store what guidance helped and what didn’t. Helpful tips are ready for the next run, and you can plug in Postgres when you want a durable trail of persona memories.
- **Production Telemetry & Export** – Out of the box you get a terminal feed that shows lane decisions, probe confidence, certification flags, and reward scores. Export the same telemetry to JSONL with one CLI call (`arc-atlas`) so training pipelines can consume it without extra wiring.
- **Bring-Your-Own-Agent Harness** – Point the SDK at whatever agent you already run, OpenAI-compatible chat, a Python function, or an HTTP endpoint. Drop your prompts and tools into the provided YAML templates and the runtime handles the rest.
- **Lightweight Defaults** – Your first run doesn’t spin up databases or exporters. All the heavier integrations (storage, dashboards, advanced telemetry) stay optional until you explicitly enable them.

---

## Quick Start

**1. Install the SDK**
```bash
pip install arc-atlas
```

**2. Configure your API keys**
```bash
export OPENAI_API_KEY=sk-...
export GOOGLE_API_KEY=...
```

**3. Run an Example**

Create a python file `run_quickstart.py`:
```python
from atlas import core

result = core.run(
    task="Summarise the latest Atlas SDK updates",
    config_path="configs/examples/openai_agent.yaml", # Assumes you cloned the repo
)

print(result.final_answer)
```

Then run the script:
```bash
python run_quickstart.py
```

---

## 📚 Full Documentation

The README hits the highlights. For the complete guide—including configuration tables, orchestration deep dives, and training recipes—visit [docs.arc.computer](https://docs.arc.computer).

---

## Architecture

![Atlas SDK Adaptive Runtime](public/runtime-2.png)

```
1. core.run()                 # load config, adapter, execution context
2. planner role creates plan  # BYOA bridge composes dependency-aware steps
3. validator role reviews     # ensures tooling, dependencies, and risks are handled
4. Orchestrator.arun()        # executes steps, applies guidance, records telemetry
5. Evaluator.ajudge()         # aggregates reward signals (process/helpfulness/custom)
6. Database.log_*()           # optional persistence of plans, attempts, trajectory events
```

Trajectory events stream through `ExecutionContext.event_stream`, enabling live console streaming and durable storage via `atlas/runtime/storage/database.py` and `atlas/runtime/storage/schema.sql`.

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
- `atlas` builds the SDK image, installs the package, and runs the ARC demo entrypoint by default (see `docker/entrypoint.sh`).
- Pass a custom command to run other configs:  
  `docker compose -f docker/docker-compose.yaml run --rm atlas python -m atlas.cli.main --help`

The container mounts your repo at `/workspace`, so you can edit code locally and rerun without rebuilding. The default entrypoint is `docker/entrypoint.sh`; override it by supplying arguments after the service name (they replace the demo command).

---

## Using `pip install arc-atlas`

When you install the SDK from PyPI you still need a PostgreSQL URL if you want persistence. The CLI now ships with a helper that can prepare a local Postgres for you:

```bash
pip install arc-atlas
# Option A – use Docker (recommended)
atlas storage up  # writes atlas-postgres.yaml and starts the container

# Option B – run docker compose yourself if you prefer
docker compose -f docker/docker-compose.yaml up -d postgres

# Either export these for the current shell or ensure they're present in .env
export STORAGE__DATABASE_URL=postgresql://atlas:atlas@localhost:5433/atlas
export OPENAI_API_KEY=sk-...
# Optional Process/Helpfulness judges
export GOOGLE_API_KEY=...

# Minimal runner script example (save as run_atlas.py)
# -----------------------------------------------
# from atlas import core
#
# result = core.run(
#     task="Summarise the Atlas SDK",
#     config_path="path/to/config.yaml",
#     stream_progress=True,
# )
# print(result.final_answer)
#
# Then execute:
# python run_atlas.py
```
- `atlas storage up` requires Docker on PATH; it writes `atlas-postgres.yaml` and runs `docker compose -f atlas-postgres.yaml up -d postgres` behind the scenes. If Docker is unavailable, the command prints the exact compose invocation to run manually.
- The compose configuration exposes Postgres on host port `5433`; keep the URL in sync if you change the mapping.
- You can point `storage.database_url` inside your YAML config or rely on the `STORAGE__DATABASE_URL` environment variable shown above.
- If storage is optional for your workflow, set `storage: null` in the config—runs will skip persistence but still execute end-to-end.
- No Docker? Install Postgres by hand (local package manager, managed instance, etc.) and point `STORAGE__DATABASE_URL` at that server instead.

---

## Exporting Runtime Sessions

Atlas persists full execution traces whenever PostgreSQL storage is configured. Convert those sessions into training-ready
JSONL with the bundled exporter:

```bash
# 1. Run tasks that log to Postgres (configure storage.database_url in your AtlasConfig)
atlas.core.run(...)

# 2. Export the captured sessions to JSONL (use the unique CLI name to avoid PATH collisions)
arc-atlas --database-url postgresql://localhost:5433/atlas --output traces.jsonl --limit 25

# (Fall back to python -m if shell PATH ordering is unpredictable)
python -m atlas.cli.export --database-url postgresql://localhost:5433/atlas --output traces.jsonl --limit 25

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
| `student` | Limits and prompt templates for the planning / execution / synthesis roles that drive your agent |
| `teacher` | Parameters for the validation and guidance role (LLM settings, cache behaviour, prompt overrides) |
| `orchestration` | Retry policy, per-step timeout, and trajectory emission flags |
| `rim` | Judge definitions, weights, aggregation strategy, thresholds |
| `storage` | Optional PostgreSQL connection info for persistence |

Atlas ships opinionated prompt templates for three cooperative roles:

1. **Planner** – drafts a dependency-aware plan that sequences tools and actions.
2. **Executor** – carries out each step and produces structured outputs (status, artifacts, deliverables).
3. **Validator / Guide** – inspects execution, supplies corrective guidance, and triggers certification rewards when needed.

Override the defaults by providing explicit `student.prompts` and `teacher.prompts` blocks in your configuration. You can tailor each role’s prompt text directly—no `{base_prompt}` substitution required—while keeping token budgets and retry settings consistent.

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
  Atlas SDK ships an adaptive dual-agent reasoning harness...
- gather dataset A | attempts: 1 | score: 0.91
- synthesise findings | attempts: 2 | score: 0.88
RIM scores | max: 0.91 | avg: 0.89
```

Disable streaming with `core.run(..., stream_progress=False)` when piping output or running in CI. Pass `stream_progress=True` to force streaming even when stdout is not a TTY. The renderer also works with `core.arun` and runs alongside PostgreSQL persistence, so stored sessions retain full telemetry.

See `docs/examples/terminal_telemetry.md` for a step-by-step walkthrough.

For a deeper look at how these events map onto the Atlas training stack—and why the SDK keeps telemetry lightweight—see
`docs/runtime_telemetry_overview.md`.

---

## Exporting Runtime Sessions

When persistence is enabled, every run captures plans, telemetry, and reward data. Convert those sessions into JSONL with the `arc-atlas` CLI:

```bash
arc-atlas \
  --database-url postgresql://atlas:atlas@localhost:5433/atlas \
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
  "final_answer": "The SDK routes BYOA agents through an adaptive dual-agent reasoning loop guided by rewards...",
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

Each exported step embeds the original executor text along with `metadata.structured_output`, so you can extract fields like `status` or `artifacts` directly from that JSON payload. Examples live in `docs/examples/export_runtime_traces.md`.

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
- Reactive stream helpers live under `atlas/utils/reactive/`; SPDX headers are retained and must remain intact.
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
