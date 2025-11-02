# Scripts Directory

This directory contains utility scripts for evaluating, validating, and analyzing Atlas SDK performance. These tools help you:

- **Validate** that features are working correctly
- **Benchmark** different model configurations
- **Report** on learning and performance metrics
- **Collect** and export datasets for evaluation

All scripts follow a consistent naming convention based on their purpose, making it easy to find the right tool for your needs.

## Script Categories

### Validation Scripts (`validate_*`)

Use these scripts to verify that Atlas SDK features are working correctly in your environment:

- **`validate_digest_stats.py`** - Verifies that prompt digest_stats telemetry are captured correctly in `.atlas/runs` artifacts
- **`validate_tool_adoption.py`** - Verifies that tool adoption tracking works end-to-end with production-style adapters
- **`validate_configs.py`** - Validates learning configuration across all config files
- **`validate_learning_metrics.py`** - Validates learning state from database (metrics, pruning, transfer success)

### Benchmarking Scripts (`benchmark_*`)

Compare model performance and configurations across different scenarios:

- **`benchmark_reward_models.py`** - Evaluates reward judge pairings against captured session trajectories
- **`benchmark_dual_agent_models.py`** - Evaluates dual-agent (student/teacher) model pairings on synthetic runtime tasks
- **`benchmark_probe_models.py`** - Evaluates capability probe models across a dataset

### Reporting Scripts (`report_*`)

Generate reports from your Atlas SDK telemetry data:

- **`report_learning.py`** - Generates learning evaluation reports from persisted telemetry (queries Postgres directly)

### Data Collection Scripts (`collect_*`, `export_*`)

Gather and export datasets for evaluation and analysis:

- **`collect_reward_trajectories.py`** - Collects SessionTrajectory payloads prior to reward scoring for evaluation datasets
- **`export_probe_dataset.py`** - Exports capability probe evaluation dataset from Postgres sessions

## Usage Examples

### Validation

Run validation scripts to ensure your Atlas SDK setup is working correctly:

```bash
# Validate digest stats capture
python scripts/validate_digest_stats.py --config configs/eval/learning/tool_adoption_claude.yaml

# Validate learning metrics
python scripts/validate_learning_metrics.py <learning_key> [--config <config_path>]
```

### Benchmarking

Compare different model configurations to find optimal settings:

```bash
# Benchmark reward models
python -m scripts.benchmark_reward_models \
  --dataset atlas/data/reward_eval_trajectories.jsonl \
  --output results/reward/eval.json

# Benchmark dual-agent models
python -m scripts.benchmark_dual_agent_models \
  --dataset atlas/data/synthetic_runtime_tasks.jsonl \
  --output results/dual_agent_eval.json
```

### Reporting

Generate reports from your telemetry data:

```bash
# Generate learning report
python scripts/report_learning.py \
  --database-url postgresql://atlas:atlas@localhost:5433/atlas \
  --learning-key <key>
```

### Data Collection

Collect and export datasets for evaluation:

```bash
# Collect reward trajectories
python -m scripts.collect_reward_trajectories \
  --tasks atlas/data/synthetic_runtime_tasks.jsonl \
  --output atlas/data/reward_eval_trajectories.jsonl \
  --limit 30

# Export probe dataset
python -m scripts.export_probe_dataset \
  --database-url $DATABASE_URL \
  --output data/probe_eval.jsonl
```

## Naming Convention

Scripts use a consistent prefix pattern based on their purpose:

- **`validate_*`** - Tests/verifies functionality
- **`benchmark_*`** - Performance/accuracy evaluation
- **`report_*`** - Report generation from data
- **`collect_*`** - Data collection
- **`export_*`** - Data export

This naming convention makes it easy to find scripts by purpose and understand their role at a glance.

## Getting Started

If you're new to Atlas SDK evaluation:

1. Start with **validation scripts** to ensure your setup is working correctly
2. Use **benchmarking scripts** to compare different model configurations
3. Generate **reports** to analyze your telemetry data
4. Use **data collection scripts** to build evaluation datasets

For detailed documentation on evaluation workflows, see:
- [Learning Evaluation](docs/evaluation/learning_eval.md)
- [Reward Evaluation](docs/evaluation/reward_eval.md)
- [Runtime Evaluation](docs/evaluation/runtime_eval.md)
- [Probe Evaluation](docs/evaluation/probe_eval.md)

