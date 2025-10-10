# Atlas SDK — PyPI Quickstart

Atlas wraps your Bring-Your-Own-Agent (BYOA) in a guided Teacher → Student → Reward loop. Install the SDK from PyPI, point it at your agent, and Atlas handles planning, orchestration, evaluation, and optional persistence for you.

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
export OPENAI_API_KEY=sk-...
# Optional judges (only if you enable the RIM helpers)
export GOOGLE_API_KEY=...
```

Atlas reads additional provider keys from adapter-specific `llm.api_key_env` fields.

## Create a Minimal Config

Save the following as `atlas_quickstart.yaml` (storage disabled by default):

```yaml
agent:
  type: openai
  name: quickstart-openai-agent
  system_prompt: |
    You are the Atlas Student. Follow instructions carefully and keep responses concise.
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
  active_judges:
    process: false
    helpfulness: false
storage: null
```

## Run Your First Task

```python
from atlas import core

result = core.run(
    task="Summarise the latest Atlas SDK updates",
    config_path="atlas_quickstart.yaml",
)

print(result.final_answer)
```

`result` is an `atlas.types.Result` containing the final answer, reviewed plan, and per-step evaluations. Set `stream_progress=True` to mirror planner/executor telemetry in your terminal.

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
docker compose -f docker/docker-compose.yaml up -d postgres

export STORAGE__DATABASE_URL=postgresql://atlas:atlas@localhost:5433/atlas_arc_demo
```

Add a `storage` section to your config when you want Atlas to log plans, attempts, and telemetry into Postgres for later inspection.

## Observe and Export

- Set `stream_progress=True` in `core.run` to stream planner/executor/judge events.
- Export stored sessions with `arc-atlas --database-url postgresql://... --output traces.jsonl`.
- Explore `docs/examples/` for telemetry and export walkthroughs.

## Next Steps

- Browse `configs/examples/` for richer orchestration templates.
- Enable RIM judges by toggling `rim.active_judges`.
- Integrate Atlas into async services with `core.arun`.
