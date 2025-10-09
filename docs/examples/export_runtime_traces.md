# Exporting Runtime Traces

The Atlas SDK ships with a standalone JSONL exporter so you can transform persisted runtime sessions into trainer-friendly datasets.

## Prerequisites

- A PostgreSQL database populated by running `atlas.core.run` with `storage.database_url` configured.
- `asyncpg` installed (included with the SDK dependencies).
- File system access to write the exported JSONL file.

## Basic Usage

```bash
arc-atlas \
  --database-url postgresql://atlas:atlas@localhost:5432/atlas \
  --output traces.jsonl

# Deterministic fallback that ignores shell PATH ordering
python -m atlas.cli.export \
  --database-url postgresql://atlas:atlas@localhost:5432/atlas \
  --output traces.jsonl
```

The command connects to the configured database, loads every stored session (or a filtered subset), and emits newline-delimited JSON. Friendly progress logs report how many sessions and steps were exported. Compatibility aliases `atlas.export` / `atlas-export` still resolve to the same CLI, but `arc-atlas` (or the `python -m atlas.cli.export` form) avoids collisions with other tools named `atlas`.

### Useful Flags

- `--session-id 42 --session-id 43` – export only the specified session IDs.
- `--limit 200 --offset 100` – page through large datasets.
- `--batch-size 250` – tune the paging size for the `sessions` table.
- `--trajectory-limit 2000` – override the number of intermediate telemetry events captured per session.

## JSONL Schema

Each line in `traces.jsonl` is an `AtlasSessionTrace`. The layout mirrors the dataclasses in `trainers/runtime_dataset.py` (`AtlasSessionTrace`, `AtlasStepTrace`, `AtlasRewardBreakdown`):

- `task` – original task prompt.
- `final_answer` – synthesised response returned by the Student.
- `plan` – reviewed plan as stored in PostgreSQL.
- `steps` – array of `AtlasStepTrace` objects with:
  - `step_id`, `description`, `tool`, and `tool_params` from the plan.
  - `trace`, `output`, and `attempts` from runtime execution.
  - `reward` captured as an `AtlasRewardBreakdown` (score, judges, samples).
  - `validation` results from the Teacher and `guidance` history.
  - `context.prior_results` containing outputs from prior steps.
  - `metadata` with retry attempt payloads and dependency hints.
- `session_metadata` – includes persisted metadata, execution status, timestamps, and the ordered list of trajectory events recorded during orchestration.

## Training Workflow

The exported file slots directly into the Atlas core training stack:

1. Run `atlas.core.run(...)` to generate sessions and persist them to PostgreSQL.
2. Execute `arc-atlas --database-url ... --output traces.jsonl` (or `python -m atlas.cli.export ...`).
3. In the core repository, call `load_runtime_traces("traces.jsonl")` or `flatten_traces_for_training(...)` from `trainers/runtime_dataset.py`.

This workflow keeps runtime telemetry decoupled from training data generation while reusing the shared schema consumed by the core trainers.
