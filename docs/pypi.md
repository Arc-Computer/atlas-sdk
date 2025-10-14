# Atlas SDK — PyPI Quickstart

Atlas wraps your Bring-Your-Own-Agent (BYOA) in a guided Teacher → Student → Reward loop. Install the SDK from PyPI, point it at your agent, and Atlas handles planning, orchestration, evaluation, and optional persistence for you.

> Atlas defaults to an in-memory workflow—leave `storage: null` in your config for quick experiments. You can add PostgreSQL later if you want durable telemetry.

## What's New in v0.1.3

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
export GOOGLE_API_KEY=... # for reward system
```

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
    provider: google
    model: gemini/gemini-2.5-flash
    api_key_env: GOOGLE_API_KEY
    max_output_tokens: 8096
  large_model:
    provider: google
    model: gemini/gemini-2.5-flash
    api_key_env: GOOGLE_API_KEY
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
# Start a local Postgres via Docker (requires Docker Desktop/Engine)
atlas storage up  # writes atlas-postgres.yaml and starts the container

# Or run docker compose yourself if you prefer:
# docker compose -f docker/docker-compose.yaml up -d postgres

# Point Atlas at the database
export STORAGE__DATABASE_URL=postgresql://atlas:atlas@localhost:5433/atlas
```

Add a `storage` section to your config when you want Atlas to log plans, attempts, and telemetry into Postgres for later inspection. If Docker isn’t available, install Postgres manually and provide the same connection URL.

## Observe and Export

- Set `stream_progress=True` in `core.run` to stream planner/executor/judge events alongside the adaptive summary.
- Export stored sessions with `arc-atlas --database-url postgresql://... --output traces.jsonl`—the JSONL includes `adaptive_summary`, `session_reward`, learning notes, and persona usage.
- Explore `docs/examples/` for telemetry and export walkthroughs.

## Next Steps

- Browse `configs/examples/` for richer orchestration templates.
- Enable RIM judges by toggling `rim.active_judges`.
- Integrate Atlas into async services with `core.arun`.
