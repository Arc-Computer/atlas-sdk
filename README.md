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

Trajectory events stream through `ExecutionContext.event_stream`, enabling live console streaming and durable storage via `atlas/storage/database.py` and `atlas/storage/schema.sql`.

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
