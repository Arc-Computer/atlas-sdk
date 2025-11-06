# Atlas SDK
[![Atlas SDK hero](public/atlas-sdk.jpeg)](public/atlas-sdk.jpeg)

[![PyPI version](https://img.shields.io/pypi/v/arc-atlas.svg)](https://pypi.org/project/arc-atlas/)
[![Downloads](https://static.pepy.tech/badge/arc-atlas)](https://pepy.tech/project/arc-atlas)
[![Python Versions](https://img.shields.io/pypi/pyversions/arc-atlas.svg)](https://pypi.org/project/arc-atlas/)
[![arXiv](https://img.shields.io/badge/arXiv-2511.01093-b31b1b.svg)](https://arxiv.org/abs/2511.01093)
[![Docs](https://img.shields.io/badge/Docs-latest-green)](https://docs.arc.computer)

Atlas is a system for continual learning from agent workflows. This repository is the runtime component—it wraps existing agents, captures execution traces with reward signals, and exports structured data for training. [Atlas Core](https://github.com/Arc-Computer/ATLAS) is the training component—it runs GRPO, GKD, and SFT on those exports to produce improved teacher checkpoints. Together they form a closed loop: the runtime generates training data from agent execution, Core trains better models from that data, you deploy updated checkpoints back into the runtime.

## How the System Works

Atlas separates runtime orchestration from offline training. This repository handles data collection, Core handles model improvement.

**Runtime (Atlas SDK):**
- Wraps your agent (OpenAI, Claude, Gemini, local models, custom implementations) in a dual-agent loop where Student executes and Teacher supervises
- Routes tasks to auto/paired/coach supervision lanes based on a capability probe that assesses difficulty and confidence
- Captures execution traces: plans, attempts, interventions, rewards at step and session granularity
- Stores telemetry in Postgres with review gates for approved sessions

**Training ([Atlas Core](https://github.com/Arc-Computer/ATLAS)):**
- Reads runtime data directly from Postgres via `atlas/training_data/`
- Trains teacher models using GRPO (RL from rewards), GKD (distillation), or SFT (supervised fine-tuning)
- Shares reward infrastructure with the runtime so scoring is consistent across data collection and training
- Produces checkpoints that deploy back into the SDK

The training algorithm itself—GRPO is a single equation over logprobs—is straightforward. The challenge is infrastructure: collecting clean training data from multi-turn agent execution with proper reward attribution, adaptive supervision, and export guardrails. That's what this SDK does.

## What Problem This Solves

If you're experimenting with RL for LLM agents, you need training data that captures more than prompt/completion pairs. You need execution traces showing where reasoning failed, how supervision corrected it, and which strategies worked. You need rewards attributed to specific steps so GRPO can learn what actions improve outcomes. You need this data exportable with review workflows so bad episodes don't poison training datasets.

Building that infrastructure means solving:
- Multi-turn orchestration with tool calls and state management
- Adaptive supervision routing (when to guide vs. when to let the agent run)
- Reward attribution across process quality, outcome correctness, and efficiency
- Telemetry persistence with plan/step/trajectory granularity
- Export guardrails with approval gates and drift detection

The SDK implements that infrastructure so you can focus on training experiments. See [`examples/mcp_tool_learning/`](examples/mcp_tool_learning/README.md) for a working integration with LangGraph agents demonstrating progressive learning across 25 file operation tasks.

---

## Runtime Features

- **Automated Configuration Discovery** – `atlas env init` scans your codebase for agent classes and tool schemas, generates runtime config, and synthesizes adapter factories when needed. See [Configuration Guide](docs/configs/configuration.md) for details.
- **Adaptive Supervision Routing** – Capability probe routes tasks to auto/paired/coach lanes based on difficulty and confidence, reducing supervision overhead as models improve on specific task types.
- **Reward Attribution** – Small/large judge pairs score process quality, outcome correctness, and efficiency at step and session granularity. Reward infrastructure is shared with Atlas Core for scoring consistency.
- **Observability and Telemetry** – Runtime sessions stream to Postgres with plan structures, execution traces, and reward payloads. Learning reports (`scripts/report_learning.py`) filter by project/task/tags and break down performance metrics.
- **Export Guardrails** – Session exports default to approved-only with CLI review workflow and drift alerts. Prevents problematic episodes from entering training datasets.
- **Direct Training Integration** – `atlas train` exports sessions and launches Atlas Core training with Hydra config overrides, closing the runtime→training loop in one command.

---

## Quick Start

> **Note**: Use Python 3.10 or newer before installing. Pip on older interpreters (e.g., 3.9) resolves `arc-atlas` 0.1.0 and the runtime crashes at import time.

**Install and onboard in three commands:**

```bash
pip install arc-atlas
atlas env init
atlas run --config .atlas/generated_config.yaml --task "Your task here"
```

**What happens:**

1. **Install** – Install the SDK from PyPI
2. **Autodiscovery** – `atlas env init` scans your codebase for environment and agent classes, analyzes their structure, and generates a runtime configuration. If no Atlas-ready classes are found, it synthesizes lightweight wrapper factories using LLM-assisted code analysis.
3. **Run** – `atlas run` executes your agent with the generated config, streams adaptive telemetry, and saves traces to `.atlas/runs/`.

The generated files (`.atlas/generated_config.yaml`, `.atlas/generated_factories.py`, `.atlas/discover.json`) are repo-aware and mirror your project's prompts, tools, and LLM choices. See [Autodiscovery Guide](docs/guides/introduction.mdx) for details.

### Prerequisites

- Python 3.10+ (3.13 recommended)
- LLM credentials exported (`OPENAI_API_KEY`, `GEMINI_API_KEY`, etc.) or present in a `.env` file

**Storage (required for rewards and learning):**

The SDK works without persistent storage but requires PostgreSQL to store reward signals and learning playbooks. Choose one:

```bash
# Option 1: Local Postgres via Docker (recommended for getting started)
atlas init

# Option 2: Add Postgres connection to your config.yaml
storage:
  database_url: postgresql://user:pass@host:port/database
```

Without storage, the SDK runs but rewards and learning history are not persisted.

### Try the Quickstart Demo

For a hands-on demonstration of Atlas learning capabilities:

```bash
atlas quickstart
```

This runs 3 security review tasks showing learning progression. See [Quickstart Guide](docs/sdk/quickstart.mdx) for detailed usage.

---

## Examples

- [`examples/mcp_tool_learning/`](examples/mcp_tool_learning/README.md) - MCP tool learning with LangGraph agents, demonstrating progressive learning across 25 file operation tasks
- `atlas quickstart` - Runs 3 security review tasks showing learning progression ([Quickstart Guide](docs/sdk/quickstart.mdx))

## Documentation

**Configuration:**
- [Configuration Guide](docs/configs/configuration.md) - Student/teacher/reward system configuration, learning tuning, adaptive teaching
- [docs.arc.computer](https://docs.arc.computer) - Full reference including orchestration details and training recipes

**Evaluation:**
- [Learning Evaluation](docs/evaluation/learning_eval.md) - Transfer learning metrics, baseline comparison, evaluation harness
- [Runtime Evaluation](docs/evaluation/runtime_eval.md) - Dual-agent runtime benchmarking and performance analysis
- [Reward Evaluation](docs/evaluation/reward_eval.md) - Judge scoring matrices and reward model validation
- [Probe Evaluation](docs/evaluation/probe_eval.md) - Capability probe accuracy and supervision routing analysis

**Operations:**
- [Export Guardrails](docs/operations/guardrails.md) - Session review, approval workflow, drift detection

<details>
<summary>Video: Installation and Configuration Walkthrough</summary>

<video src="public/Atlas.sdk-high.mp4" controls width="100%">
  Your browser does not support the video tag. <a href="public/Atlas.sdk-high.mp4">Download the video</a>.
</video>

</details>

---

## Architecture

![Atlas SDK Adaptive Runtime](public/runtime-2.png)

```
1. core.run()                 # load config, adapter, execution context
2. Student planner creates plan  # Bring-Your-Own-Agent bridge composes dependency-aware steps
3. Teacher validator reviews     # ensures tooling, dependencies, and risks are handled
4. Capability probe selects supervision lane  # routes to auto, paired, or coach based on confidence
5. Orchestrator.arun()        # executes steps, applies guidance, records telemetry
6. Evaluator.ajudge()         # aggregates reward signals (process/helpfulness/custom)
7. Database.log_*()           # stores plans, attempts, trajectory events in Postgres
8. Review + export guards     # reward stats + drift alerts gate training exports until approved
```
---

## Configuration

Configuration files live in `configs/examples/`. Each YAML document is validated against `atlas.config.models.AtlasConfig`.

**Quick reference of configuration sections:**

| Section | Purpose |
| ------- | ------- |
| `agent` | Adapter settings (endpoint, Python import path, model) and tool schemas |
| `student` | Prompts and limits for Student persona (planner, executor, synthesizer) |
| `teacher` | Teacher persona settings (LLM config, cache behavior, prompts) |
| `orchestration` | Retry policy, per-step timeout, trajectory emission |
| `rim` | Judge models, weights, aggregation strategy, thresholds |
| `adaptive_teaching` | Capability probe, supervision lane thresholds, learning history |
| `storage` | PostgreSQL connection info for persistence |

See the [Configuration Guide](docs/configs/configuration.md) for detailed tuning options including learning synthesis, reward system configuration, and adaptive teaching parameters.

---

## Training Data Access

Training workflows require persistent storage to capture reward signals and execution traces. The runtime uses PostgreSQL for persistence.

**Setup:**

```bash
# Option 1: Local Postgres via Docker
atlas init  # Starts bundled Docker + Postgres on localhost:5433

# Option 2: Use your own Postgres instance (add to config.yaml)
storage:
  database_url: postgresql://user:pass@host:port/database
```

Once storage is configured, runtime sessions stream to the database automatically. Atlas Core accesses this data directly:

```python
from atlas.training_data import get_training_sessions

sessions = get_training_sessions(
    db_url="postgresql://atlas:atlas@localhost:5433/atlas",
    min_reward=0.7,
    review_status_filters=["approved"],
    limit=100
)
```

**Optional: JSONL export**

For offline workflows or external tools, export sessions to JSONL:

```bash
arc-atlas \
  --database-url postgresql://atlas:atlas@localhost:5433/atlas \
  --include-status approved \
  --output traces.jsonl \
  --limit 100
```

Each line is an `AtlasSessionTrace` with plans, steps, rewards, and metadata. See `docs/examples/export_runtime_traces.md` for details.

---

## Training Your Model

Once you've collected runtime traces, use [Atlas Core](https://github.com/Arc-Computer/ATLAS) to train updated teacher models.

**Training Methods:**
- **GRPO** - Reinforcement learning from reward signals ([Guide](https://docs.arc.computer/training/offline/grpo-training))
- **GKD** - 9-30x faster distillation for production models ([Guide](https://docs.arc.computer/training/offline/gkd-training))
- **SFT** - Supervised fine-tuning on approved traces

**Quick Start:**

```bash
# Option 1: Direct database access (recommended)
export STORAGE__DATABASE_URL=postgresql://atlas:atlas@localhost:5433/atlas
export ATLAS_CORE_PATH=~/src/ATLAS

atlas train \
  --config-name offline/base \
  --trainer-config grpo \
  --wandb-project atlas-runtime \
  --override trainer.max_steps=250

# Option 2: Export to JSONL first
arc-atlas --database-url postgresql://... --output traces.jsonl
cd $ATLAS_CORE_PATH
python scripts/run_offline_pipeline.py --export-path traces.jsonl
```

**Deploying Trained Models:**

After training, update your SDK config to use the improved teacher:

```yaml
# config.yaml - HuggingFace Inference Endpoint
teacher:
  llm:
    provider: openai  # HF inference is OpenAI-compatible
    model: your-org/atlas-teacher-v1
    api_base: https://api-inference.huggingface.co/models/your-org/atlas-teacher-v1
    api_key_env: HUGGING_FACE_HUB_TOKEN
    temperature: 0.05

# config.yaml - Local inference server (vLLM/TGI)
teacher:
  llm:
    provider: openai  # Most local servers are OpenAI-compatible
    model: your-org/atlas-teacher-v1
    api_base: http://localhost:8000/v1
    api_key_env: VLLM_API_KEY  # Dummy key if server doesn't require auth
    temperature: 0.05
```

Run agents with the improved teacher to collect better training data, creating a continual learning loop.

**Comprehensive Guides:**
- [Complete Training Pipeline](https://docs.arc.computer/training/offline/grpo-training) - Step-by-step SFT → GRPO workflow
- [Training Configuration](https://docs.arc.computer/training/configuration) - Hydra parameters reference
- [Training Data Pipeline](https://docs.arc.computer/training/offline/training-data-pipeline) - Direct database access API

---

## Testing

```bash
PYTHONPATH=. pytest tests --disable-warnings
```

The test suite covers dependency parsing, prompt rewriting, student/teacher orchestration, reward system aggregation, adapter bridges, and database logging. Most tests rely on locally mocked adapters, so no external network calls occur.

For evaluation harnesses (runtime, reward, learning, probe), see the [Evaluation documentation](#documentation) above.

---

## Requirements & Notes

- Python 3.10+ (project is developed and validated with 3.13).
- Development extras (`pip install -e .[dev]`) install pytest tooling for local validation; core telemetry streams rely solely on the standard library.
- Reactive stream helpers live under `atlas/utils/reactive/`; SPDX headers are retained and must remain intact.
- Aim for descriptive naming and concise docstrings so the intent is evident without extra commentary.

---

## Development

```bash
# Install with dev dependencies
pip install -e .[dev]

# Run tests
PYTHONPATH=. pytest tests --disable-warnings

# Format and lint
ruff check .
ruff format .

# Type checking (if pyright is installed)
pyright
```

For evaluation harnesses (runtime, reward, learning, probe), see the [Evaluation documentation](#documentation) above.

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
