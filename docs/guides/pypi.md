# Atlas SDK — PyPI Quickstart

Atlas wraps your Bring-Your-Own-Agent (BYOA) in a guided Teacher → Student → Reward loop. Install the SDK from PyPI, point it at your agent, and Atlas handles planning, orchestration, evaluation, and optional persistence for you.

> Atlas defaults to an in-memory workflow—leave `storage: null` in your config for quick experiments. You can add PostgreSQL later if you want durable telemetry.

## What's New in v0.1.8

- **Autodiscovery & CLI Upgrades** – `atlas env init` now scaffolds full configs, auto-loads `.env`/`PYTHONPATH`, and can replay discoveries with `atlas run --config` or the fake LLM smoke-test path (`ATLAS_FAKE_LLM=1`) to validate stacks offline ([#52](https://github.com/Arc-Computer/atlas-sdk/pull/52), [#70](https://github.com/Arc-Computer/atlas-sdk/pull/70), [#74](https://github.com/Arc-Computer/atlas-sdk/pull/74), [#75](https://github.com/Arc-Computer/atlas-sdk/pull/75)).
- **Learning Playbooks in Runtime** – Student and Teacher personas fetch hashed “learning playbooks”, inject them into every planner/synthesizer/executor prompt, and track metadata so cached prompts stay in sync when playbooks change ([#76](https://github.com/Arc-Computer/atlas-sdk/pull/76)).
- **Persistent Telemetry & Learning Reports** – Discovery and runtime sessions log directly to Postgres, and the new learning evaluation harness can filter by project/task/tags while generating model-level breakdowns in Markdown/JSON reports ([#72](https://github.com/Arc-Computer/atlas-sdk/pull/72), [#73](https://github.com/Arc-Computer/atlas-sdk/pull/73)).
- **Safety Guardrails & Approvals** – Session exports require explicit approval, with CLI tooling to review/approve/quarantine runs and drift alerts captured alongside reward metadata ([#63](https://github.com/Arc-Computer/atlas-sdk/pull/63)).
- **Expanded Evaluation Suites** – Added capability probe updates (xAI Grok support), dual-agent runtime benchmarking, and a reward model harness with packaged datasets and docs to keep offline validation comprehensive ([#55](https://github.com/Arc-Computer/atlas-sdk/pull/55), [#56](https://github.com/Arc-Computer/atlas-sdk/pull/56), [#57](https://github.com/Arc-Computer/atlas-sdk/pull/57)).
- **Lean Learning History Payloads** – Capability probe history now respects an operator-defined cap, trims noisy fields, and keeps streak stats lightweight for faster probes ([#54](https://github.com/Arc-Computer/atlas-sdk/pull/54)).

## What's New in v0.1.7

- **Adaptive Runtime** – Capability probe selects execution mode (`auto`, `paired`, `coach`, `escalate`) per request based on task complexity and historical performance.
- **Persistent Learning Memory** – Guidance from each episode is tagged by reward and automatically reused on similar tasks.
- **Fingerprint-Based Certification** – First-run tasks get certified, enabling auto mode on future similar requests when confidence is high.

## Install in Minutes

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install arc-atlas
```

- Python 3.10 or newer is required (3.13 recommended).
- For development tooling and tests, install extras with `pip install arc-atlas[dev]`.

## Configure Your Environment

Set API keys before running Atlas:

```bash
export OPENAI_API_KEY=sk-... # your api key
export GEMINI_API_KEY=... # for reward system
```

Prefer storing secrets in a `.env` file? The SDK automatically loads it on startup (via `python-dotenv`), so CLI commands and examples pick up those values without manual exports.

Atlas reads additional provider keys from adapter-specific `llm.api_key_env` fields.

## Create a Minimal Config

Save the following as `atlas_quickstart.yaml` (storage disabled by default):

```yaml
agent:
  type: openai
  name: quickstart-openai-agent
  system_prompt: |
    You are an Agent. Follow instructions carefully and keep responses concise.
  tools: []
  llm:
    provider: openai
    model: gpt-4o-mini
    api_key_env: OPENAI_API_KEY
    temperature: 0.1
    max_output_tokens: 1024
student:
  max_plan_tokens: 1024
  max_step_tokens: 1024
  max_synthesis_tokens: 1024
teacher:
  llm:
    provider: openai
    model: gpt-4o-mini
    api_key_env: OPENAI_API_KEY
    temperature: 0.1
    max_output_tokens: 768
orchestration:
  max_retries: 1
  step_timeout_seconds: 600
  emit_intermediate_steps: true
rim:
  small_model:
    provider: gemini
    model: gemini/gemini-2.5-flash
    api_key_env: GEMINI_API_KEY
    max_output_tokens: 8096
  large_model:
    provider: gemini
    model: gemini/gemini-2.5-flash
    api_key_env: GEMINI_API_KEY
    max_output_tokens: 8096
  judge_prompt: 'reward the agent for attending the issues mentioned in the task'
  variance_threshold: 0.15
  uncertainty_threshold: 0.3
storage: null
```

## Run Your First Task

```python
from atlas import core

result = core.run(
    task="Summarise the latest Atlas SDK updates",
    config_path="atlas_quickstart.yaml",
    stream_progress=True,
)

print(result.final_answer)
```

`result` is an `atlas.types.Result` containing the final answer, reviewed plan, and per-step evaluations. Set `stream_progress=True` to mirror planner/executor telemetry in your terminal.
The console summary includes the adaptive mode, confidence, certification flag, and session reward so you can watch the J-curve without any database setup.

Need the structured metadata? Access `ExecutionContext.get().metadata` after the run or export later via the CLI once storage is configured.

## Wrap Your Existing Agent

### OpenAI-Compatible Chat Agent

```python
from atlas import core
from atlas.connectors import create_adapter
from atlas.config.models import OpenAIAdapterConfig

adapter = create_adapter(OpenAIAdapterConfig(
    type="openai",
    name="my-openai-agent",
    system_prompt="You are a helpful assistant.",
    tools=[],
    llm={
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
    },
))

result = core.run(
    task="Draft a product brief for Atlas",
    config_path="atlas_quickstart.yaml",
    adapter_override=adapter,
)
```

Override the adapter to reuse the same orchestration settings with different agents.

### Local Python Function

```python
# my_agent.py
def respond(prompt: str, metadata: dict | None = None) -> str:
    return f"echo: {prompt}"
```

Update the config’s `agent` block:

```yaml
agent:
  type: python
  name: local-function-agent
  system_prompt: |
    You call a local Python function named respond.
  import_path: my_agent
  attribute: respond
  tools: []
```

Atlas imports your callable (optionally from `working_directory`), handles async execution, generator outputs, and metadata passing.

### HTTP Endpoint

```yaml
agent:
  type: http_api
  name: http-agent
  system_prompt: |
    You delegate work to a REST endpoint that accepts {"prompt": "..."}.
  transport:
    base_url: https://your-agent.example.com/v1/atlas
    timeout_seconds: 60
  payload_template:
    prompt: "{{ prompt }}"
  result_path: ["data", "output"]
  tools:
    - name: web_search
      description: Search the web.
      parameters:
        type: object
        properties:
          query:
            type: string
        required: [query]
```

Atlas retries requests based on the adapter’s `retry` policy and normalises JSON responses using `result_path`.

## Optional: Persist Runs with PostgreSQL

```bash
# Start a local Postgres via Docker (installs Docker if missing)
atlas init  # writes atlas-postgres.yaml, starts Postgres, and applies the schema

# Or run docker compose yourself if you prefer:
# docker compose -f docker/docker-compose.yaml up -d postgres

# Point Atlas at the database
export STORAGE__DATABASE_URL=postgresql://atlas:atlas@localhost:5433/atlas
```

Add a `storage` section to your config when you want Atlas to log plans, attempts, and telemetry into Postgres for later inspection. If Docker isn’t available, install Postgres manually and provide the same connection URL.

## Observe and Export

- Set `stream_progress=True` in `core.run` to stream planner/executor/judge events alongside the adaptive summary.
- Export stored sessions with `arc-atlas --database-url postgresql://... --output traces.jsonl`—the JSONL includes `adaptive_summary`, `session_reward`, per-session learning notes, the consolidated `learning_state`, and aggregated history.
- Explore `docs/examples/` for telemetry and export walkthroughs.

## Train with Atlas Core

Use the SDK CLI to bridge runtime traces into the Atlas Core training pipeline:

```bash
git clone https://github.com/Arc-Computer/ATLAS ~/src/ATLAS
export ATLAS_CORE_PATH=~/src/ATLAS
export STORAGE__DATABASE_URL=postgresql://atlas:atlas@localhost:5433/atlas

atlas train --config-name offline/base --dry-run
# inspect the command, then rerun without --dry-run to execute training
```

`atlas train` writes a JSONL export to `<atlas-core-path>/exports/<timestamp>.jsonl` and then executes `scripts/run_offline_pipeline.py` from that directory. You can point `--output` at a custom path, forward Hydra overrides with repeated `--override` flags, or use `--output-dir` / `--wandb-project` to steer checkpoints and logging. Pass `--use-sample-dataset` to copy the bundled sample dataset when you just want to validate the workflow without hitting Postgres.

## Next Steps

- Browse `configs/examples/` for richer orchestration templates.
- Enable RIM judges by toggling `rim.active_judges`.
- Integrate Atlas into async services with `core.arun`.
