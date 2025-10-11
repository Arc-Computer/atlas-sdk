# Streaming SRE Continual-Learning Demo

This demo packages the end-to-end workflow described in `STREAMING_SRE_DEMO_PLAN.md`. It streams production-style incidents, routes them through the Atlas runtime with persona memories enabled, visualises adaptation in real time, and exports the trajectory data that would be used for downstream fine-tuning.

The runtime makes **real OpenAI API calls** (`gpt-5` by default) so you can showcase the actual learning loop rather than a mocked agent, while keeping prompts lean via a triage stage that mirrors production systems like incident.io.

## Prerequisites

- Python 3.10+ with Atlas SDK dependencies installed (`pip install -e .[dev]`)
- Docker (to start the demo Postgres instance)
- OpenAI API key with access to `gpt-5`
- Optional: local Postgres if you prefer not to use Docker compose

## Setup

1. Create a virtual environment and install Atlas dependencies if you have not already:

   ```bash
   python3.13 -m venv .venv
   source .venv/bin/activate
   pip install -e .[dev]
   ```

2. Copy the demo environment template and populate it:

   ```bash
   cp examples/streaming_sre_demo/.env.example .env
   ```

   Required entries:

   - `OPENAI_API_KEY` – key with `gpt-5` access
   - `ATLAS_SRE_DEMO_DATABASE_URL` – Postgres DSN (defaults to the compose stack on `localhost:5433`)
   - Optional overrides: `SRE_AGENT_MODEL`, `SRE_AGENT_TEMPERATURE`

3. Start the demo Postgres instance:

   ```bash
   make demo-db
   ```

   This runs `docker compose -f docker/docker-compose.yaml up -d postgres`.

## Running the Demo

Open two terminals after activating your virtual environment.

### Terminal 1 – live incident feed

```bash
python -m examples.streaming_sre_demo.data_stream --speed slow
```

This prints Datadog-style logs for each incident at the chosen cadence.

### Terminal 2 – Atlas streaming driver

```bash
python -m examples.streaming_sre_demo.run_streaming_demo --speed slow
```

What you’ll see:

- A Rich table that updates per incident with execution mode, attempts, reward, token usage deltas, and applied persona memories.
- A side panel logging persona-memory lifecycle events (candidate creation, promotions, demotions).
- Automatic JSONL exports per incident under `examples/streaming_sre_demo/exports/jsonl/`.
- CSV roll-up (`demo_results.csv`) for quick plotting or blog charts.

`--speed fast` and `--speed replay` accelerate the cadence. Use `--skip-export` if you want to dry-run without Postgres.

### What Atlas receives

`run_streaming_demo.py` now performs a **triage pass** before calling `atlas.core.run`:

- Synthesises an incident summary (severity, service, primary symptom).
- Attaches key metrics, the most recent deploy/change, and related historical incidents.
- Includes a runbook hint that mimics incident.io’s “what to do next” guidance.
- Trims the raw log excerpt to the top two entries so the LLM stays within budget.

The triaged payload is serialised to JSON and becomes the task passed to Atlas. This keeps prompts under ~800 tokens while preserving the high-signal context that the persona-learning loop needs.

### Persona-memory reporting helper

After a run you can inspect the accumulated memories:

```bash
python -m examples.streaming_sre_demo.reporting --tenant-id sre-demo
```

Optionally filter by persona:

```bash
python -m examples.streaming_sre_demo.reporting --persona student_executor --persona teacher_validation
```

## Make Targets

| Command            | Description                                             |
|--------------------|---------------------------------------------------------|
| `make demo-db`     | Start the Postgres container used by the demo           |
| `make demo`        | Start Postgres then launch the streaming driver         |
| `make demo-stream` | Tail the incident generator in “slow” mode              |
| `make demo-clean`  | Stop containers and remove Postgres volumes             |

## Data Flow & Exports

- **Triage-first incidents**: `examples/streaming_sre_demo/data_stream.py` emits reproducible incidents (including novel mTLS failures at IDs 5 and 15) enriched with metrics, deploy metadata, and similar incidents—mirroring incident.io’s “full picture” feed.
- **Driver**: `run_streaming_demo.py` triages each payload, calls `atlas.core.run`, clears the persona cache between runs, captures telemetry, and writes exports.
- **Exports**:
  - JSONL traces per incident for replay/fine-tuning (`exports/jsonl/incident_XXXX.jsonl`)
  - Tabular summary (`exports/demo_results.csv`)
- **Reporting helper**: `reporting.py` queries `persona_memory` and `persona_memory_usage` for before/after comparisons.

## Telemetry Expectations

1. Incident 5 (`novel_mtls`) should fail on the first attempt, requiring teacher guidance and generating a persona-memory candidate grounded in the runbook hint.
2. Between incidents the promotion loop runs; the cache is cleared so the next fingerprint fetch reflects any promotions.
3. Incident 15 (`novel_mtls`) should succeed on the first attempt with a higher reward, lower retries, and reduced tokens. The table shows a negative Δ tokens percentage; the events panel announces the promotion and the triaged payload shows the previously-learned memory in use.

## Presenter Script (Suggested Talking Points)

1. **Cold start:** Highlight the first few incidents where the system handles known failures without persona memory assistance. Point out the baseline rewards and retries in the table.
2. **Novel outage (Incident #5):** Show the failing execution, low reward, and the memory event announcing a new candidate. Mention the JSONL export capturing the failed trajectory plus teacher guidance.
3. **Promotion moment:** Explain that Atlas evaluates the candidate against reward/retry deltas and promotes it, visible in the events panel (`promotion XXXX → active`). Tie it back to the triage summary so the audience understands how the memory applies before the incident is re-run.
4. **Adaptation proof (Incident #15):** Emphasise the successful first-attempt diagnosis, higher reward, and the negative token delta, demonstrating efficiency gains.
5. **Data review:** Open one of the JSONL exports or run the reporting helper to show the promoted memory and its usage stats.

## Troubleshooting

- If the driver cannot reach Postgres, confirm `make demo-db` is running or update `ATLAS_SRE_DEMO_DATABASE_URL`.
- If OpenAI returns quota errors, verify the key and product entitlements for `gpt-5`.
- Use `--skip-export` to run without Postgres (persona-memory persistence will be disabled).
- Run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -m "not postgres"` before committing changes.

## Next Steps

- Swap in real incident payloads by pointing the generator at your telemetry source.
- Extend the reporting helper to emit dashboards (e.g., Streamlit or Grafana).
- Feed the JSONL exports into your fine-tuning or evaluation pipeline.
