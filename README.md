# Atlas SDK

Atlas SDK lets you wrap any Bring-Your-Own-Agent (BYOA) into a guided Teacher → Student → Reward loop. The toolkit focuses on sequential, high-trust workflows: you supply an HTTP endpoint, a Python function, or an OpenAI-compatible agent; Atlas handles planning, orchestration, evaluation, and persistence.

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
pip install -e .[dev,dashboard]
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
| `prompt_rewrite` | LLM used to derive planner / executor / teacher personas from the user prompt |

During startup Atlas calls the rewrite LLM once to transform the BYOA system prompt into three personas:

1. **Planner Student** – drafts a dependency-aware plan
2. **Executor Student** – runs each step and returns a trace
3. **Teacher** – reviews plans, validates execution, and issues retries/guidance

By default the rewrite call reuses the same API credentials as your agent. Provide an explicit `prompt_rewrite` block if
you need a dedicated model or different limits.

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

Trajectory events stream through `ExecutionContext.event_stream`, enabling live dashboards or durable storage via `atlas/storage/database.py` and `atlas/storage/schema.sql`.

**RIM Model Guidance**

- Tier-1 judges (process/helpfulness): Gemini 2.5 Flash or Grok-4 Fast provide fast, low-cost scores.
- Tier-2 arbiter: Gemini 2.5 Pro reconciles disagreements with high fidelity.
- Supplied examples show how to point `rim.judges[].llm` and `rim.arbiter` at different providers if desired.

---

## Telemetry Dashboard

The dashboard runs as an optional FastAPI service that renders live telemetry and stored runs.

1. Provision PostgreSQL:

   ```bash
   docker compose -f docker-compose.dashboard.yml up -d
   export ATLAS_DATABASE_URL="postgresql://atlas:atlas@localhost:5432/atlas"
   psql "$ATLAS_DATABASE_URL" -f atlas/storage/schema.sql
   ```

2. Start the dashboard service:

   ```bash
   python -m atlas.dashboard --database-url "$ATLAS_DATABASE_URL"
   ```

3. In another terminal, execute the sample agent with telemetry enabled:

   ```bash
   python examples/telemetry_dashboard_demo.py --database-url "$ATLAS_DATABASE_URL" \
     --task "Summarize the Atlas SDK architecture"
   ```

4. Visit `http://127.0.0.1:8000` to browse sessions, timelines, evaluations, and live events.

Tear down the compose stack when finished:

```bash
docker compose -f docker-compose.dashboard.yml down
```

---

## GDPval Demo

GDPval is a validation benchmark that pairs GDP growth tasks with curated reference documents.

1. Install extras and export credentials:

   ```bash
   pip install -e .[dev,dashboard,gdpval]
   export OPENAI_API_KEY=...
   # Optional: export HUGGINGFACEHUB_API_TOKEN if your datasets auth is restricted
   ```

2. Start PostgreSQL and the telemetry dashboard (see the Telemetry Dashboard section above).

3. Run a single GDPval task:

   ```bash
   python examples/gdpval/run_gdpval.py --task-id gdpval_task_001 --config configs/examples/gdpval_python.yaml
   ```

   Cached references are stored in `.atlas/gdpval/<task-id>/` and surface in the dashboard metadata panel.

4. Stream the full split (cap execution with `--max` when needed):

   ```bash
   python examples/gdpval/run_gdpval.py --all --max 10 --config configs/examples/gdpval_python.yaml
   ```

   Summaries accumulate at `examples/gdpval/gdpval_runs/` as `runs.csv` and `runs.jsonl`.

5. Open `http://127.0.0.1:8000` to filter sessions by sector or occupation, inspect cached references, and observe live telemetry while tasks execute.

---

## Testing

```bash
PYTHONPATH=. pytest tests --disable-warnings
```

The suite covers dependency parsing, prompt rewriting, student/teacher orchestration, RIM aggregation, adapter bridges, and database logging. Most tests rely on locally mocked adapters, so no external network calls occur.

---

## Requirements & Notes

- Python 3.10+ (project is developed and validated with 3.13).
- Optional dependencies (installed via `pip install -e .[dev,dashboard,gdpval]`) include `litellm`, `langchain-core`, `langgraph`, `asyncpg`, the FastAPI dashboard stack, and GDPval helpers (`datasets`, `pypdf`, `python-docx`).
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
