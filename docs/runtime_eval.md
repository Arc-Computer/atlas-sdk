## Dual-Agent Runtime Evaluation

Use the dual-agent evaluation harness to compare student/teacher model pairings against a consistent set of synthetic runtime tasks. The CLI keeps all changes outside the core runtime so you can iterate on model selection without touching orchestrator logic.

### Goals
- Validate default student/teacher pairings across SRE, security, compliance, customer comms, and analytics scenarios.
- Capture latency, reward, and lightweight answer-quality indicators for each pairing.
- Produce a machine-readable summary to inform the default model selection.

### Dataset
- Location: `atlas/data/synthetic_runtime_tasks.jsonl`.
- Format: newline-delimited JSON with keys:
  - `task` – natural-language persona-aligned prompt.
  - `expected_answer` – short string used for similarity scoring.
  - `metadata` – dictionary for tags such as `scenario` and `difficulty`.
- The first line is a README-style comment describing how to extend the file. Append new tasks by adding JSON lines that follow the same schema.

### Model Matrix & Credentials

| Role    | Model Identifier              | Provider            | Required API key env var |
|---------|------------------------------|---------------------|--------------------------|
| Student | `gpt-5-mini`                 | OpenAI (`openai`)   | `OPENAI_API_KEY`         |
| Student | `claude-haiku-4-5`           | Anthropic           | `ANTHROPIC_API_KEY`      |
| Student | `gemini-2.5-flash`           | Gemini              | `GEMINI_API_KEY`         |
| Student | `grok-4-fast`                | xAI                 | `XAI_API_KEY`            |
| Teacher | `gpt-5`                      | OpenAI (`openai`)   | `OPENAI_API_KEY`         |
| Teacher | `claude-sonnet-4-5-20250929` | Anthropic           | `ANTHROPIC_API_KEY`      |
| Teacher | `gemini-2.5-pro`             | Gemini              | `GEMINI_API_KEY`         |
| Teacher | `grok-4-fast`                | xAI                 | `XAI_API_KEY`            |

Populate the relevant keys in your shell or `.env` file; the script calls `load_dotenv_if_available()` before resolving model parameters.

### Running the Harness
```bash
python -m scripts.eval_dual_agent_models \
  --dataset atlas/data/synthetic_runtime_tasks.jsonl \
  --output results/dual_agent_eval.json
```

Key options:
- `--student-models` / `--teacher-models` – space-separated lists of LiteLLM identifiers to evaluate.
- `--base-config` – alternate runtime config to clone for overrides (defaults to `configs/examples/openai_agent.yaml`).
- `--repeats` – rerun each task N times (useful for sampling variance).
- `--concurrency` – maximum in-flight executions (defaults to sequential).
- `--similarity-threshold` – tweak the difflib ratio required to mark a task as matched.
- `--output` – write a JSON artifact containing both per-run records and aggregated summaries.

Advanced overrides: set `ATLAS_MODEL_OVERRIDE_<MODEL_ID>` (e.g. `ATLAS_MODEL_OVERRIDE_GPT_5=azure/gpt-5-preview`) to point a specific identifier at a custom endpoint while reusing the same provider defaults. Use `ATLAS_MODEL_TIMEOUT` to raise/lower the per-call timeout across all models.

### Metrics Captured
Per task:
- Final answer returned by `atlas.core.run`.
- Adaptive mode and mode history derived from `ExecutionContext.metadata["adaptive_summary"]`.
- Session reward score when present.
- Wall-clock runtime and exception flag.
- Similarity score and boolean match flag vs. `expected_answer`.

Aggregated per model pair:
- Average runtime and reward (successes only).
- Accuracy (share of runs that exceed the similarity threshold).
- Failure count and adaptive-mode distribution.

CLI output renders a summary table and, when `--output` is provided, persists a JSON report with the full run history and a computed “best pair” record.
