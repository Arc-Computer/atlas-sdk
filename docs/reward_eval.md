## Reward Model Evaluation

This evaluation mirrors the probe and runtime sweeps for reward interpreters (RIM). The harness replays captured session trajectories, swaps judge pairings, and reports how each combination scores, escalates, and performs. Use it to validate default reward models and justify alternates before rolling them into production configs.

### Objectives
- Compare small/large judge stacks against the dataset that ships with `configs/examples/openai_agent.yaml`.
- Track average reward, variance, escalation frequency, latency, and correlation with a baseline judge.
- Produce reproducible JSON artifacts and summary tables aligned with probe (`docs/probe_eval.md`) and runtime (`docs/runtime_eval.md`) reports.

### Dataset
- **Location:** `atlas/data/reward_eval_trajectories.jsonl`
- **Shape:** newline-delimited JSON containing a README comment followed by full `SessionTrajectory` payloads. Each object includes:
  - `task`, `final_answer`, `plan`, `steps`, `execution_mode`, `teacher_intervened`
  - `adaptive_summary`, `session_metadata`, `trajectory_type`
  - Optional `focus_prompt`
- **Composition:** four cold-start (paired lane) sessions plus two warm-start samples that include `auto` and `coach` mode traces with stored learning history. The dataset was captured by instrumenting dual-agent runs just before the runtime invoked the existing evaluator.
- **Expansion:** Use `scripts/capture_reward_trajectories.py` to record new trajectories without editing runtime code:
  ```bash
  python -m scripts.capture_reward_trajectories \
    --tasks atlas/data/synthetic_runtime_tasks.jsonl \
    --output atlas/data/reward_eval_trajectories.jsonl \
    --limit 30 \
    --repeats 2 \
    --shuffle \
    --append
  ```
  Run multiple passes (optionally with different task datasets or repeats) until at least 30 trajectories are captured; the script appends to the JSONL file while preserving the README header.

### Judge Candidates

| ID            | Small Judge (provider)              | Large Judge (provider)               | Notes |
|---------------|-------------------------------------|--------------------------------------|-------|
| `gemini_pair` | `gemini/gemini-2.5-flash` (Gemini)  | `gemini/gemini-2.5-pro` (Gemini)     | Current default shipped in `openai_agent.yaml`; used as baseline. |
| `claude_stack`| `claude-haiku-4-5` (Anthropic)      | `claude-sonnet-4-5-20250929` (Anthropic) | Claude-only stack for provider redundancy. |
| `gpt5_stack`  | `gpt-5-mini` (OpenAI)               | `gpt-5` (OpenAI)                     | Evaluates OpenAI’s small/large pairing for parity with runtime options. |
| `grok_stack`  | `xai/grok-4-fast` (xAI)             | `xai/grok-4` (xAI)                   | Tests xAI’s fast vs. larger Grok judges without cross-provider escalation. |

Add new combinations by extending `JUDGE_PRESETS` and `JUDGE_COMBOS` in `scripts/eval_reward_models.py`; no orchestrator code changes are required.

### Harness Workflow
1. Ensure the dataset is committed and up to date (`atlas/data/reward_eval_trajectories.jsonl`).
2. Export required API keys (`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `XAI_API_KEY`, etc.) and optionally store them in `.env`. The harness loads `.env` automatically via `load_dotenv_if_available()`.
3. Run the evaluator harness:
   ```bash
   python -m scripts.eval_reward_models \
     --dataset atlas/data/reward_eval_trajectories.jsonl \
     --output results/reward/eval_gemini_claude.json \
     --repeats 1 \
     --concurrency 1
   ```
4. Inspect the console table for high-level metrics and review the JSON artifact for per-run details.

Key command options:
- `--judge-combos`: space-separated combo IDs to evaluate (defaults to all built-ins).
- `--baseline`: combo ID used for deltas and correlation (defaults to `gemini_pair`).
- `--repeats`: number of passes over the dataset (helps quantify variance).
- `--concurrency`: maximum concurrent evaluations per combo.

### Metrics Captured
- **Reward score statistics:** mean and standard deviation across successful runs.
- **Uncertainty:** average model-reported uncertainty.
- **Escalation rate:** fraction of runs that triggered arbiter escalation.
- **Latency:** mean, median, and p95 wall-clock latency (milliseconds).
- **Failures:** count of judge requests that raised exceptions.
- **Baseline agreement:** delta statistics, Pearson correlation, and fraction of runs within ±0.02 reward of the baseline pair.

### Results (to be updated)

| Date       | Baseline        | Highlights | Notes |
|------------|-----------------|------------|-------|
| YYYY-MM-DD | `gemini_pair`   | _TBD_      | Run `python -m scripts.eval_reward_models` and capture JSON under `results/reward/`. |

Record final metrics in the table above once the evaluation completes, and summarize the preferred judge stack alongside escalation/latency observations. Reference the runtime findings in `docs/runtime_eval.md` to maintain continuity across probe → runtime → reward analyses.
