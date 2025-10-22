# Learning Evaluation Without Experience Hints

Atlas already captures the signals needed to explain what changed, how it changed, and why—even before hint
distillation ships. This guide documents the end-to-end workflow for analysing learning progress using the telemetry
persisted by the runtime today.

## 1. Capture Telemetry

1. **Discovery loop** – `atlas env init` records discovery telemetry per task in `discovery_runs` (Postgres) and
   `.atlas/discover.json`. The `persist_discovery_run` helper mirrors the payload in Postgres for correlation.
2. **Runtime sessions** – `atlas run` (or `atlas.core.run`) stores every session in Postgres with:
   - `sessions.metadata.learning_key` identifying the learning thread.
   - `sessions.metadata.adaptive_summary` detailing execution mode decisions.
   - `sessions.reward_stats`, `sessions.reward_audit`, and `trajectory_events` capturing reward and behavioural traces.
3. **Learning registry** – `learning_registry` keeps the latest pamphlet per `learning_key` when the synthesiser is
   enabled.

> Tip: Set `STORAGE__DATABASE_URL` before running Atlas so the runtime connects to Postgres automatically.

## 2. Export JSONL Traces (Optional)

When you need portable traces—for example to inspect raw sessions or feed downstream training—the existing exporter
already includes the signals required for hint-less evaluation:

```bash
python -m atlas.cli.export \
  --database-url postgresql://atlas:atlas@localhost:5433/atlas \
  --output results/learning/sessions.jsonl \
  --limit 200 \
  --include-status pending --include-status approved \
  --trajectory-event-limit 400
```

Each JSONL record surfaces:

- `execution_mode` (top-level + `session_metadata.adaptive_summary`)
- `learning_key`, `reward_stats`, `reward_audit`, and `session_reward`
- `trajectory_events` with `event_type` and `actor`

## 3. Run the Learning Evaluation Script

Use the new `scripts/eval_learning.py` helper to assemble structured summaries per learning key. The script queries
Postgres directly—no JSONL export required—and produces JSON + Markdown reports under `results/learning/`.

```bash
python scripts/eval_learning.py \
  --database-url postgresql://atlas:atlas@localhost:5433/atlas \
  --recent-window 10 \
  --baseline-window 50 \
  --limit 5
```

### Key options

- `--summary-only` – skips per-session trajectory fetches and relies on SQL event counts; use this for large sweeps or CI.
- `--batch-size` – max number of learning keys evaluated concurrently (default: 4).
- `--filter-project`, `--filter-task`, `--filter-tag` – narrow the run to a specific codebase, task name, or session tags.
- `--learning-key` – analyse explicit keys instead of querying Postgres for the top-N recent keys.
- `--compare-to results/learning/index.json` – diff the current run against a previous harness export; the manifest stores per-key deltas and Markdown files append a comparison section.
- `--no-markdown` – emit only machine-readable JSON for automation scenarios.

Summary mode is ideal for nightly or CI jobs where you just need reward deltas and model trends. Run the full-detail mode (default) when you want trajectory event counts sampled per session and are comfortable with additional database reads.

Outputs:

- `results/learning/<slug>_summary.json` – machine-readable payload (sessions, reward windows, discovery references).
- `results/learning/<slug>_summary.md` – human-friendly digest highlighting reward deltas, adaptive behaviour, and model breakdowns.
- `results/learning/index.json` – manifest listing every generated artifact, plus the comparison/aggregate tables when `--compare-to` is provided.

Pass `--learning-key ...` to target specific keys or `--no-markdown` when you only need JSON.

## 4. Interpret the Results

Each summary provides:

- **Reward momentum** – recent mean, baseline mean, and delta so you can spot positive/negative drift.
- **Window context** – recent/baseline window sizes so you can reason about sample counts.
- **Execution modes** – distribution of `execution_mode` values (auto, paired, coach, escalate) for the evaluated window.
- **Review state** – counts per `review_status` to ensure you compare approved vs pending runs intentionally.
- **Model performance** – per-role model breakdowns (session counts, reward averages, latest score) extracted from adapter telemetry so you can see which students/teachers are learning fastest.
- **Discovery context** – pointers to matching discovery/runtime telemetry (`discovery_runs`) for the same task so you
  can replay the original traces.
- **Latest sessions** – compact view of recent runs with reward/uncertainty snapshots and trajectory event counts.

Because everything keys off `learning_key`, you can join the summary back to:

- `sessions` (runtime telemetry + reward signals)
- `discovery_runs` (discovery/runtime captures recorded via the CLI)
- `learning_registry` (current pamphlet state, when enabled)

## 5. Compare Runs Over Time

When you pass `--compare-to`, the harness looks up the previous `index.json`, loads each saved summary, and computes deltas for:

- Reward trends (recent mean, latest score)
- Session counts per learning key
- Model-level utilisation and reward mean changes

The new manifest includes `comparisons` and aggregate leaderboards (best/worst deltas), while each Markdown report gains a “Comparison vs previous run” section.

## 6. Automate & Test

- The evaluation script ships with unit tests that stub database access and external LLM calls, so `pytest` covers the
  new entry points without touching live services.
- To keep the workflow reproducible, commit the generated summaries or re-run the script as part of your evaluation
  pipeline once fresh telemetry lands.

With these pieces in place we can meaningfully answer “what changed, how it changed, and why” today, deferring the
hint-specific analytics until the hint pipeline arrives.
