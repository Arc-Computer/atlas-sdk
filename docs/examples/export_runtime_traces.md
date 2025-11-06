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

The command connects to the configured database, loads every stored session (or a filtered subset), and emits newline-delimited JSON. By default only `review_status = approved` sessions are exported so quarantined or pending traces never leak into training by accident. Friendly progress logs report how many sessions and steps were exported. Compatibility aliases `atlas.export` / `atlas-export` still resolve to the same CLI, but `arc-atlas` (or the `python -m atlas.cli.export` form) avoids collisions with other tools named `atlas`.

### Useful Flags

- `--session-id 42 --session-id 43` – export only the specified session IDs.
- `--limit 200 --offset 100` – page through large datasets.
- `--batch-size 250` – tune the paging size for the `sessions` table.
- `--trajectory-limit 2000` – override the number of intermediate telemetry events captured per session.
- `--include-status pending` – include pending sessions alongside approved runs (repeat to add `quarantined`).
- `--include-all-statuses` – bypass the review filter entirely (use with caution).
- Environment shortcuts: `ATLAS_REVIEW_DEFAULT_EXPORT_STATUSES="approved,pending"` adjusts the default filter; `ATLAS_REVIEW_REQUIRE_APPROVAL=0` makes `pending` runs exportable without passing extra flags.

## JSONL Schema

Each line in `traces.jsonl` is an `AtlasSessionTrace`. The layout mirrors the dataclasses in `trainers/runtime_dataset.py` (`AtlasSessionTrace`, `AtlasStepTrace`, `AtlasRewardBreakdown`):

- `task` – original task prompt.
- `final_answer` – synthesised response returned by the Student.
- `plan` – reviewed plan as stored in PostgreSQL.
- `steps` – array of `AtlasStepTrace` objects with:
  - `step_id`, `description`, `tool`, and `tool_params` from the plan.
  - `trace`, `output`, and `attempts` from runtime execution.
  - `output` is a JSON string containing `status`, `artifacts`, and optional `notes`. Parse it (e.g. `json.loads(step["output"])`) to access the structured fields.
  - `reward` captured as an `AtlasRewardBreakdown` (score, judges, samples).
  - `validation` results from the Teacher and `guidance` history.
  - `context.prior_results` containing outputs from prior steps.
  - `metadata` with retry attempt payloads and dependency hints.
- `session_metadata` – includes persisted metadata, execution status, timestamps, and the ordered list of trajectory events recorded during orchestration.
- `reward_stats` – rolling statistics captured when the session reward was logged (score mean/stddev, uncertainty).
- `reward_audit` – raw prompts, responses, and reasoning metadata returned by each judge/arbiter invocation so teams can audit reward decisions.
- `review_status` / `review_notes` – human-in-the-loop guardrail metadata.
- `drift` + `drift_alert` – z-score/MAD deltas against the recent baseline for the same learning key.

## Training Workflow

The exported file slots directly into the Atlas core training stack:

1. Run `atlas.core.run(...)` to generate sessions and persist them to PostgreSQL.
2. Review new sessions with `arc-atlas review sessions --database-url ...`, then approve them with `arc-atlas review approve <id>`.
3. Execute `arc-atlas --database-url ... --output traces.jsonl` (or `python -m atlas.cli.export ...`).
3. In the core repository, call `load_runtime_traces("traces.jsonl")` or `flatten_traces_for_training(...)` from `trainers/runtime_dataset.py`.

This workflow keeps runtime telemetry decoupled from training data generation while reusing the shared schema consumed by the core trainers.
