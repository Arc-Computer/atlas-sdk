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

## Key Highlights (v0.1.10)

- **Autodiscovery-first CLI** – `atlas env init` now writes runnable configs, auto-loads `.env`/`PYTHONPATH`, and feeds straight into `atlas run --config` or the fake LLM smoke-test flow (`ATLAS_FAKE_LLM=1`) so you can validate stacks offline before hitting production creds ([docs](docs/learning_eval.md)).
- **Learning Playbooks Everywhere** – Student and Teacher personas resolve hashed learning playbooks on every run, splice the guidance into planner/synthesizer/executor prompts, and update cache keys when playbooks change so prompt drift is eliminated.
- **Persistent Telemetry & Reports** – Discovery and runtime sessions stream to Postgres, and the learning report harness filters by project/task/tags while breaking down model performance, reward deltas, and adaptive mode mix into Markdown/JSON artifacts (`scripts/eval_learning.py`).
- **Safety Guardrails for Exports** – Session exports default to approved-only with CLI review, approval, and quarantine commands plus drift alerts embedded in metadata—production pipelines stay clean while local overrides remain available.
- **Expanded Evaluation Suites** – New datasets + docs cover capability probe updates (xAI Grok), dual-agent runtime benchmarking, and reward model scoring; unit tests back each harness so you can extend with confidence.
- **Offline Training Workflow** – `atlas train` reuses export filters, ships a sample dataset, and launches Atlas Core with Hydra overrides, making the export→train loop a single CLI hop when you’re ready to fine-tune.

---

## Quick Start

> **Note**: Use Python 3.10 or newer before installing. Pip on older interpreters (e.g., 3.9) resolves `arc-atlas` 0.1.0 and the runtime crashes at import time.

### Integration at a Glance

| If your project... | Quick path | What to know |
| --- | --- | --- |
| Your repo already exposes environment and agent entrypoints Atlas can instantiate (decorators optional) | `atlas env init` → `atlas run --config .atlas/generated_config.yaml --task "..."` | `atlas env init` scans for those classes/factories, writes `.atlas/generated_config.yaml` with the actual prompts + LLM defaults, and `atlas run` streams the adaptive loop live while saving the trace under `.atlas/runs/`. |

**Prerequisites**
- `pip install arc-atlas`
- LLM credentials exported (`OPENAI_API_KEY`, `GEMINI_API_KEY`, etc.) or present in a `.env`
- Atlas needs Postgres. Run `atlas init` to start the bundled Docker + Postgres stack if you do not have one.

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

Then run the script:
```bash
atlas run --config .atlas/generated_config.yaml --task "..."
```
### Autodiscovery Onboarding

```bash
atlas env init
```

`atlas env init` scans your codebase, captures telemetry, and writes
`.atlas/discover.json` / `.atlas/generated_config.yaml`.

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
6. Database.log_*()           # stores plans, attempts, trajectory events in Postgres
7. Review + export guards     # reward stats + drift alerts gate training exports until approved
```
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
  "final_answer": "The SDK routes BYOA agents through...",
  "plan": {"steps": [...]},
  "steps": [{"step_id": 1, "reward": {"score": 0.92}, "status": "ok", "artifacts": {...}}],
  "session_metadata": {"session_id": 42, "status": "succeeded"}
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
