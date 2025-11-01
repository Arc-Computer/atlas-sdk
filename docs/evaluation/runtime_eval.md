## Dual-Agent Runtime Evaluation

Use the dual-agent evaluation harness to compare student/teacher model pairings against a consistent set of synthetic runtime tasks. The CLI keeps all changes outside the core runtime so you can iterate on model selection without touching orchestrator logic.

### Goals
- Validate default student/teacher pairings across SRE, security, compliance, customer comms, and analytics scenarios.
- Capture wall-clock latency and reward-system scores for each pairing.
- Produce a machine-readable summary to inform the default model selection.

### Dataset
- Location: `atlas/data/synthetic_runtime_tasks.jsonl`.
- Format: newline-delimited JSON with keys:
  - `task` – natural-language persona-aligned prompt.
  - `expected_answer` – short string used for similarity scoring.
  - `metadata` – dictionary for tags such as `scenario` and `difficulty`.
- The first line is a README-style comment describing how to extend the file. Append new tasks by adding JSON lines that follow the same schema.

### Model Matrix & Credentials

| Role    | Model Identifier              | Provider (LiteLLM)  | Required API key env var |
|---------|------------------------------|---------------------|--------------------------|
| Student | `gpt-5-mini`                 | OpenAI (`openai`)   | `OPENAI_API_KEY`         |
| Student | `claude-haiku-4-5`           | Anthropic           | `ANTHROPIC_API_KEY`      |
| Student | `gemini-2.5-flash`           | Gemini              | `GEMINI_API_KEY`         |
| Student | `grok-4-fast`                | xAI                 | `XAI_API_KEY`            |
| Teacher | `gpt-5`                      | OpenAI (`openai`)   | `OPENAI_API_KEY`         |
| Teacher | `claude-sonnet-4-5-20250929` | Anthropic           | `ANTHROPIC_API_KEY`      |
| Teacher | `gemini-2.5-pro`             | Gemini              | `GEMINI_API_KEY`         |
| Teacher | `grok-4-fast`                | xAI                 | `XAI_API_KEY`            |

Populate the relevant keys in your shell or `.env` file; the script calls `load_dotenv_if_available()` before resolving model parameters. When cloning the OpenAI adapter config we coerce the student provider back to `openai` under the hood so LiteLLM can route the request while satisfying the adapter validation.

### Running the Harness
```bash
python -m scripts.eval_dual_agent_models \
  --dataset atlas/data/synthetic_runtime_tasks.jsonl \
  --output results/dual_agent_eval.json
```

Key options:
- `--student-models` / `--teacher-models` – space-separated list of model identifiers (see `MODEL_PRESETS`) to evaluate.
- `--base-config` – alternate runtime config to clone for overrides (defaults to `configs/examples/openai_agent.yaml`).
- `--repeats` – rerun each task N times (useful for sampling variance).
- `--concurrency` – maximum in-flight executions (defaults to sequential). Values greater than 1 fan out work across a process pool, keeping each runtime isolated so LiteLLM logging stays stable.
- `--output` – write a JSON artifact containing both per-run records and aggregated summaries.

Advanced overrides: set `ATLAS_MODEL_OVERRIDE_<MODEL_ID>` (e.g. `ATLAS_MODEL_OVERRIDE_GPT_5=azure/gpt-5-preview`) to point a specific identifier at a custom endpoint while reusing the same provider defaults. Use `ATLAS_MODEL_TIMEOUT` to raise/lower the per-call timeout across all models.

### Metrics Captured
Per task:
- Final answer returned by the runtime.
- Adaptive mode and mode history derived from `ExecutionContext.metadata["adaptive_summary"]`.
- Session reward score when present.
- Wall-clock runtime and exception flag.

Aggregated per model pair:
- Average runtime and reward (successes only).
- Failure count and adaptive-mode distribution.

CLI output renders a summary table and, when `--output` is provided, persists a JSON report with the full run history and a computed “best pair” record.

### Learning Registry & Synthesizer
- **Reward-only judges** – Session reward prompts now emit just the scoring rationale. Student/teacher learning notes come from a dedicated synthesizer that runs after reward evaluation.
- **Learning registry** – Canonical pamphlets live in the `learning_registry` table keyed by `learning_key`. Each run still writes per-session notes to `sessions.student_learning` / `sessions.teacher_learning` for auditing.
- **Config block** – Runtime configs gain a top-level `learning` block. Use `enabled` to surface pamphlets to adapters, flip `update_enabled` off when you want “apply” mode (read-only learning), and `history_limit` to throttle how much history the synthesizer sees. Set `apply_to_prompts` to `false` if you need to disable the new playbook injection while keeping persistence intact.
- **Execution flow** – `atlas run` loads the current pamphlet into `ExecutionContext.metadata["learning_state"]` at session start. After reward evaluation, the synthesizer refreshes the pamphlet (when updates are enabled) and `Database.upsert_learning_state` persists it.

When you rerun the dual-agent harness, confirm that the emitted telemetry (e.g. `results/dual_agent_eval.json`) contains the injected "Teacher Playbook" snippets under the validation payload. This ensures the cached pamphlets influenced the teacher’s decisions during evaluation.

### Findings (2025-10-19)

We executed the full 4×4 student/teacher matrix against the 25-task synthetic dataset. Because no historical learning traces exist for these learning keys, the capability probe routed every task to the `paired` lane (student executes once, teacher validates the final answer). The table below shows the aggregated reward and latency metrics per pairing.

| Student            | Teacher                    | Avg Reward | Avg Runtime (s) | Failures | Notes |
|--------------------|----------------------------|------------|-----------------|----------|-------|
| claude-haiku-4-5   | grok-4-fast                | **0.996**  | **18.40**       | 0        | Highest reward and fastest runtime across the matrix. |
| claude-haiku-4-5   | gemini-2.5-pro             | 0.994      | 22.07           | 0        | Slightly slower, still excellent reward. |
| claude-haiku-4-5   | claude-sonnet-4-5-20250929 | 0.989      | 20.08           | 0        | All runs stayed in paired mode; no retries triggered. |
| claude-haiku-4-5   | gpt-5                      | 0.958      | 42.43           | 0        | Latency penalty without corresponding reward gains. |
| gemini-2.5-flash   | gemini-2.5-pro             | 0.990      | 25.48           | 0        | Strong reward, moderate latency. |
| gemini-2.5-flash   | grok-4-fast                | 0.954      | 22.30           | 0        | Faster but noticeably lower reward. |
| gemini-2.5-flash   | claude-sonnet-4-5-20250929 | 0.954      | 23.74           | 0        | Balanced but below the Claude student pairings. |
| gemini-2.5-flash   | gpt-5                      | 0.916      | 48.92           | 0        | Slowest in this block with the weakest reward. |
| grok-4-fast        | grok-4-fast                | 0.985      | 16.66           | 0        | Fastest absolute runtime; reward slightly behind Claude/grok. |
| grok-4-fast        | gemini-2.5-pro             | 0.989      | 20.75           | 0        | Consistent performance, modest latency. |
| grok-4-fast        | gpt-5                      | 0.988      | 42.84           | 0        | Teacher cost outweighs reward delta. |
| grok-4-fast        | claude-sonnet-4-5-20250929 | 0.991      | 19.57           | 1        | One Anthropics “overloaded” error observed; otherwise strong. |
| gpt-5-mini         | grok-4-fast                | 0.959      | 58.36           | 0        | Acceptable reward but significantly slower. |
| gpt-5-mini         | gemini-2.5-pro             | 0.960      | 67.19           | 1        | One LiteLLM request failure (“openai adapter request failed”). |
| gpt-5-mini         | claude-sonnet-4-5-20250929 | 0.929      | 66.37           | 0        | Lowest reward among Claude/Gemini teachers. |
| gpt-5-mini         | gpt-5                      | 0.937      | 98.11           | 0        | Slowest overall pairing with the weakest reward signal. |

**Key takeaways**
- **Best performing default:** `claude-haiku-4-5` as the student paired with `grok-4-fast` as the teacher delivered the highest reward (≈0.996) and low latency (~18 s). This pairing is a strong candidate for the runtime default.
- **Secondary options:** Gemini’s `2.5-pro` and Anthropic’s `claude-sonnet-4-5-20250929` teachers also performed well with the Claude and Grok students, offering provider redundancy at a small latency cost.
- **Less effective combinations:** `gpt-5-mini` as the student consistently lagged on reward and was 3–5× slower than Claude/Grok students. Coupling any student with the `gpt-5` teacher increased latency substantially without appreciable reward gains.
- **Reliability observations:** We saw two isolated failures across the matrix—one Anthropic overload and one LiteLLM “openai adapter request failed” error—both recovered on subsequent tasks. All other runs completed successfully.
- **Probe behavior:** Since the synthetic tasks had no prior learning history, the capability probe defaulted to `paired`. Seeding history or running a warm-up phase would enable evaluation of `auto`/`coach` behavior if needed.

These measurements were captured on 2025‑10‑19. Re-run the harness periodically to ensure performance remains stable as model endpoints evolve.

#### Capability Probe context

Prior to the dual-agent sweep we reran the capability-probe evaluation (see `docs/probe_eval.md`). When judging probe-only accuracy and latency, `grok-4-fast` achieved the best balance (≈80 % routing accuracy, ~2.3 s latency, with an `auto`-heavy mode mix). That result informed our choice of grok as the teacher LLM in the matrix above—paired with Claude Haiku it delivers leadership in both probe readiness and dual-agent reward. The second-best probe model, Claude Haiku 4.5 (~64 % accuracy, ~1 s latency), remains a viable failover; Gemini 2.5 Flash trails on probe accuracy (~60 %) despite decent latency.
