# Atlas SDK
[![Atlas SDK hero](public/atlas-sdk.jpeg)](public/atlas-sdk.jpeg)

[![PyPI version](https://img.shields.io/pypi/v/arc-atlas.svg)](https://pypi.org/project/arc-atlas/)
[![Downloads](https://static.pepy.tech/badge/arc-atlas)](https://pepy.tech/project/arc-atlas)
[![Python Versions](https://img.shields.io/pypi/pyversions/arc-atlas.svg)](https://pypi.org/project/arc-atlas/)

The Atlas SDK is a drop-in learning harness that enables your agent to learn from experience, adapt to new challenges, and become more efficient over time - all without modifying your existing agent code or weights. It wraps any agent (OpenAI, Claude, Gemini, local models, or your own stack) with an adaptive dual-agent reasoning loop guided by reward signals, so agents stay fast on familiar work while escalating supervision on new or risky tasks. The SDK records rich telemetry, surfaces adaptive signals in real time, and exports production data for downstream training.

> **How it relates to [ATLAS](https://github.com/Arc-Computer/ATLAS)**  
> This repository delivers the runtime harness that powers continual learning in production. The `ATLAS` repo focuses on training models that ingest the structured traces produced here. Run the SDK to capture adaptive episodes; feed those traces into ATLAS to retrain and evaluate new policies.

---

With the split between SDK (runtime) and ATLAS (training) in mind, here's what our runtime gives you out of the box.

## Key Highlights (v0.1.8)

- **Autodiscovery-first CLI** – `atlas env init` now writes runnable configs, auto-loads `.env`/`PYTHONPATH`, and feeds straight into `atlas run --config` or the fake LLM smoke-test flow (`ATLAS_FAKE_LLM=1`) so you can validate stacks offline before hitting production creds ([docs](docs/learning_eval.md)).
- **Learning Playbooks Everywhere** – Student and Teacher personas resolve hashed learning playbooks on every run, splice the guidance into planner/synthesizer/executor prompts, and update cache keys when playbooks change so prompt drift is eliminated.
- **Persistent Telemetry & Reports** – Discovery and runtime sessions stream to Postgres, and the learning report harness filters by project/task/tags while breaking down model performance, reward deltas, and adaptive mode mix into Markdown/JSON artifacts (`scripts/eval_learning.py`).
- **Safety Guardrails for Exports** – Session exports default to approved-only with CLI review, approval, and quarantine commands plus drift alerts embedded in metadata—production pipelines stay clean while local overrides remain available.
- **Expanded Evaluation Suites** – New datasets + docs cover capability probe updates (xAI Grok), dual-agent runtime benchmarking, and reward model scoring; unit tests back each harness so you can extend with confidence.
- **Offline Training Workflow** – `atlas train` reuses export filters, ships a sample dataset, and launches Atlas Core with Hydra overrides, making the export→train loop a single CLI hop when you’re ready to fine-tune.

---

## Quick Start

<Note>
Use Python 3.10 or newer before installing. Pip on older interpreters (e.g., 3.9) resolves `arc-atlas` 0.1.0 and the runtime crashes at import time.
</Note>

### Integration at a Glance

| If your project... | Quick path | What to know |
| --- | --- | --- |
| Already exposes Atlas-shaped classes (`reset`/`step`/`close`, `plan`/`act`/`summarize`) or uses `@atlas.environment` / `@atlas.agent` decorators | `atlas env init --task "..."` → `atlas run --task "..."` | Discovery hashes your modules, writes `.atlas/discover.json` + `.atlas/generated_config.yaml`, and you’re ready to go—no factories required. |
| Relies on LangGraph, SecRL, or other frameworks that need custom constructors/config | `atlas env scaffold --template langgraph` (optional helper), then `atlas env init --env-fn ... --agent-fn ...` | Use lightweight factory helpers (see `examples/langgraph_adapter.py`) to instantiate your stack. Pass extra args via `--env-arg/--agent-arg` and configs via `--env-config/--agent-config`. |

**Prerequisites**
- `pip install arc-atlas`
- LLM credentials exported (`OPENAI_API_KEY`, `GEMINI_API_KEY`, etc.) or present in a `.env`
- Optional but recommended for learning persistence: `STORAGE__DATABASE_URL=postgresql://atlas:atlas@localhost:5433/atlas` (run `atlas init` to scaffold Dockerised Postgres if needed)

**1. Create a virtual environment & install the SDK**
```bash
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install arc-atlas
```

**2. Configure your API keys**
```bash
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=...
export XAI_API_KEY=...
```
Keys can also live in a local `.env` file; the Atlas CLI and quickstart scripts automatically load it via [python-dotenv](https://pypi.org/project/python-dotenv/).

If you want persistent learning updates, ensure a Postgres instance is reachable (the `atlas init` helper will start one via Docker) and export `STORAGE__DATABASE_URL` before running the CLI.

**3. Run an Example**

Create a python file `run_quickstart.py`:

```python
from atlas import core

result = core.run(
    task="Summarise the latest Atlas SDK updates",
    config_path="configs/examples/openai_agent.yaml",
)

print(result.final_answer)
```

Then run the script:
```bash
python run_quickstart.py
```

> **Tip for local development:** the runtime now gates exports on approved sessions. If you are experimenting locally and don’t want to run the review CLI every time, set `ATLAS_REVIEW_REQUIRE_APPROVAL=0` in your shell before exporting traces. Production deployments should keep approval enabled.

### Autodiscovery Onboarding

Pick the flow that matches your project:

**Decorator-first (no factories)**

```bash
pip install arc-atlas
# Ensure your environment / agent classes are decorated with @atlas.environment / @atlas.agent
atlas env init --task "Investigate production incident"
atlas run --task "Investigate production incident"
```

**Factory-driven (custom stacks)**

```bash
pip install arc-atlas
# Optionally scaffold a reference implementation into your repo
atlas env scaffold --template langgraph

atlas env init --task "Investigate production incident" \
  --env-fn langgraph_adapter:create_environment \
  --agent-fn langgraph_adapter:create_agent \
  --env-arg incident_id=38 --no-run
atlas run --task "Investigate production incident"
```

`atlas env init` scans your codebase, captures telemetry, and writes
`.atlas/discover.json` / `.atlas/generated_config.yaml`. Pass `--scaffold-config-full`
to have the CLI merge discovery metadata into a runnable Atlas configuration using
the OpenAI example template—provider/model hints are propagated automatically when
we can introspect your agent stack. When it detects heavier stacks
(LangGraph/DeepAgents, multi-service environments, etc.) it records preflight notes and skips
execution until you rerun with `--validate`. When autodiscovery finds no candidates it prints a reminder pointing to
`atlas env scaffold --template langgraph` so you can start from a working factory.
Discovery now auto-loads `.env` (including `PYTHONPATH`) before spinning up your
factories and will derive additional import hints from the factories you pass on
the command line—most stacks no longer need manual `PYTHONPATH` exports. For
offline validation pipelines, set `ATLAS_FAKE_LLM=1` to run smoke tests against
the generated config without calling a live provider.

See [`examples/langgraph_adapter.py`](examples/langgraph_adapter.py) and the [Stateful Agent Quickstart](docs/sdk/quickstart.mdx)
for end-to-end templates, configuration snippets, and troubleshooting tips.

---

## 📹 Video Walkthrough

<details>
<summary><b>Watch: Complete Installation & Configuration Guide</b> (click to expand)</summary>

<br>

This video provides a complete walkthrough of installing the Atlas SDK and configuring your first agent.

<video src="public/Atlas.sdk-high.mp4" controls width="100%">
  Your browser does not support the video tag. <a href="public/Atlas.sdk-high.mp4">Download the video</a>.
</video>

</details>

---

## 📚 Full Documentation

The README hits the highlights. For the complete guide—including configuration tables, orchestration deep dives, and training recipes—visit [docs.arc.computer](https://docs.arc.computer). For the hint-less learning evaluation workflow, see `docs/learning_eval.md`.

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
7. Review + export guards     # reward stats + drift alerts gate training exports until approved
```

Trajectory events stream through `ExecutionContext.event_stream`, enabling live console streaming and durable storage via `atlas/runtime/storage/database.py` and `atlas/runtime/storage/schema.sql`. Every persisted session now carries `reward_stats`, a `drift` payload, and a `review_status` (`pending`, `approved`, or `quarantined`). The `arc-atlas review ...` commands surface drift deltas and let operators approve or quarantine traces before `arc-atlas` exports them (approved runs remain the default filter).

Add a `runtime_safety` block to tune guardrails without touching code:

```yaml
runtime_safety:
  drift:
    window: 75            # number of prior sessions in the baseline
    z_threshold: 2.5      # sensitivity for score/uncertainty drift
    min_baseline: 10      # minimum history before alerts fire
  review:
    require_approval: true
    default_export_statuses: ["approved"]
```

Environment knobs (`ATLAS_DRIFT_WINDOW`, `ATLAS_DRIFT_Z_THRESHOLD`, `ATLAS_DRIFT_MIN_BASELINE`, `ATLAS_REVIEW_DEFAULT_EXPORT_STATUSES`, `ATLAS_REVIEW_REQUIRE_APPROVAL`) override the YAML for quick local experiments.

---

## Run with Docker

The repo ships with a ready-to-go Compose stack under `docker/`:

```bash
# 1. Ensure your project .env includes the required keys (Compose reads it automatically):
#    OPENAI_API_KEY=sk-...
#    GEMINI_API_KEY=...
# 2. Build the SDK image and start Postgres + the demo agent
docker compose -f docker/docker-compose.yaml up --build
```

- `postgres` starts a local PostgreSQL instance with a persisted volume (`atlas_pg_data`).
- `atlas` builds the SDK image, installs the package, and runs the ARC demo entrypoint by default (see `docker/entrypoint.sh`). The entrypoint uses `ATLAS_QUICKSTART_CONFIG=docker/configs/atlas.docker.yaml`, which expects the Compose-provided Postgres service.
- Compose reads `.env` automatically, so the container sees the same `OPENAI_API_KEY` / `GEMINI_API_KEY` values you use locally.
- Pass a custom command to run other configs:  
  `docker compose -f docker/docker-compose.yaml run --rm atlas python -m atlas.cli.main --help`

The container mounts your repo at `/workspace`, so you can edit code locally and rerun without rebuilding. The default entrypoint is `docker/entrypoint.sh`; override it by supplying arguments after the service name (they replace the demo command).

---

## Using `pip install arc-atlas`

When you install the SDK from PyPI you still need a PostgreSQL URL if you want persistence. The CLI now ships with a helper that can prepare a local Postgres for you:

```bash
pip install arc-atlas
# Option A – use Docker (recommended)
atlas init  # installs Docker if missing, writes atlas-postgres.yaml, and starts Postgres

# Option B – run docker compose yourself if you prefer
docker compose -f docker/docker-compose.yaml up -d postgres

# Either export these for the current shell or ensure they're present in .env
export STORAGE__DATABASE_URL=postgresql://atlas:atlas@localhost:5433/atlas
export OPENAI_API_KEY=sk-...
# Optional Process/Helpfulness judges
export GEMINI_API_KEY=...
export XAI_API_KEY=...

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
- `atlas init` installs Docker when possible, writes `atlas-postgres.yaml`, starts the PostgreSQL container, and applies the Atlas schema automatically.
- The compose configuration exposes Postgres on host port `5433`; keep the URL in sync if you change the mapping.
- You can point `storage.database_url` inside your YAML config or rely on the `STORAGE__DATABASE_URL` environment variable shown above.
- Shut everything down with `atlas quit` (use `--purge` to remove the Docker volume) when you no longer need local storage.
- If storage is optional for your workflow, set `storage: null` in the config—runs will skip persistence but still execute end-to-end.
- No Docker? Install Postgres by hand (local package manager, managed instance, etc.) and point `STORAGE__DATABASE_URL` at that server instead—or run `atlas init --skip-docker-install` to reuse an existing Docker Engine.

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
| `adaptive_teaching` | Capability probe defaults, persistent-learning history limit, and reward objectives |
| `storage` | Optional PostgreSQL connection info for persistence |

> `adaptive_teaching.learning_history_limit` controls how many recent sessions are surfaced to the capability probe.
> It defaults to 10 (max 200). Override it in YAML under the `adaptive_teaching` block, or set the
> `ATLAS_LEARNING_HISTORY_LIMIT` environment variable for a temporary change (env overrides the config when present).

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

## Runtime → Training

Once you have traces in Postgres you can hand them to the Atlas Core training stack without writing glue scripts. The SDK now ships `atlas train`, which exports sessions to JSONL and calls `scripts/run_offline_pipeline.py` inside your Atlas Core clone.

**Prerequisites**
- Clone [Arc-Computer/ATLAS](https://github.com/Arc-Computer/ATLAS) and set `ATLAS_CORE_PATH` (or pass `--atlas-core-path`).
- Provide a Postgres URL via `STORAGE__DATABASE_URL` or `DATABASE_URL` when exporting live data.
- Ensure your Python environment has the dependencies required by Atlas Core (see its README).

With those in place you can launch a training run end-to-end:

```bash
export STORAGE__DATABASE_URL=postgresql://atlas:atlas@localhost:5433/atlas
export ATLAS_CORE_PATH=~/src/ATLAS

atlas train \
  --config-name offline/base \
  --trainer-config trainer/openai \
  --wandb-project atlas-runtime \
  --override trainer.max_steps=250
```

The command writes a timestamped export to `<ATLAS_CORE_PATH>/exports/`, then executes Atlas Core from within that directory. Pass `--output` to control the JSONL location, `--output-dir` to steer Hydra’s checkpoint directory, or repeatable `--override` flags for custom Hydra overrides. Use `--dry-run` to preview the exact invocation without running training, or `--use-sample-dataset` to copy the bundled sample dataset when you just want to validate wiring.

On success you will see the export path echoed back along with a reminder that Atlas Core checkpoints land under `<atlas-core-path>/outputs` unless overridden.

---

## Testing

- Dual-agent runtime evaluation harness: see `docs/runtime_eval.md` for metrics, dataset schema, and CLI usage.
- Reward model evaluation harness: see `docs/reward_eval.md` for judge matrices, dataset schema, and replay commands.

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
