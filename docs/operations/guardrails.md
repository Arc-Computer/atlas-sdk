# Quarantine & Drift Workflow

Atlas now enforces a human-in-the-loop review step before session traces reach downstream training loops. Every persisted session carries an explicit `review_status`:

- `pending` – default for freshly captured sessions.
- `approved` – ready for export and training.
- `quarantined` – held out until an operator clears the drift or scoring issue.

Sessions remain pending until a reviewer promotes them. Reward drift detection runs immediately after a session is logged; if a new reward deviates more than three standard deviations (or MAD-equivalent) from the recent baseline for the same learning key, the session metadata gains a `drift` block and the `drift_alert` flag flips to `true`.

```json
"reward_stats": {
  "score": 0.84,
  "score_stddev": 0.03,
  "sample_count": 3,
  "uncertainty_mean": 0.11,
  "timestamp": "2025-03-01T18:44:05Z"
},
"reward_audit": [
  {
    "stage": "tier1",
    "model": "gpt-4o-mini",
    "temperature": 0.2,
    "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
    "response": "{\"score\": 0.84, \"uncertainty\": 0.12}",
    "reasoning_queue": [...]
  }
],
"drift": {
  "drift_alert": true,
  "score_delta": 0.42,
  "uncertainty_delta": 0.18,
  "reason": "score_z"
}
```

## Inspecting the quarantine queue

Use the new admin subcommands to review sessions directly from the database:

```bash
arc-atlas review sessions --database-url postgresql://localhost:5432/atlas
```

The output groups sessions by review status and surfaces drift metrics (`scoreΔ`, `uncΔ`) alongside the reward snapshot. To zoom in on a specific queue:

```bash
arc-atlas review sessions --status pending --database-url ...
```

## Approving or quarantining sessions

Once you have inspected a trace, promote or quarantine it with an optional note:

```bash
arc-atlas review approve 123 --note "Reward validated in notebook" --database-url ...
arc-atlas review quarantine 124 --note "Instructor score disagrees" --database-url ...
```

Notes are stored in `review_notes`, while the drift payload remains in `session_metadata["drift"]` for audit trails.

## Exporting only vetted traces

`arc-atlas` now exports **only approved sessions**. Pending or quarantined runs are ignored unless you explicitly request them:

```bash
# Default: approved only
arc-atlas --database-url ... --output traces.jsonl

# Include additional queues when triaging
arc-atlas --database-url ... --output traces.jsonl --include-status pending

# Export everything (use with care)
arc-atlas --database-url ... --output traces.jsonl --include-all-statuses
```

Each JSONL line now includes `review_status`, `review_notes`, `reward_stats`, and the `drift` block so downstream trainers can perform their own safety checks.

## Operational checklist

1. **Monitor** `arc-atlas review sessions` for new drift alerts.
2. **Investigate** flagged traces – the enriched JSON includes drift deltas, reward baselines, and learning keys.
3. **Decide**: approve or quarantine with context in `--note`.
4. **Export** approved traces; quarantine stays out of training until cleared.

This guardrail keeps humans in the loop and stops anomalous rewards from seeding continual-learning updates without oversight.

## Tuning guardrails

Guardrails are configurable through YAML and environment overrides:

```yaml
runtime_safety:
  drift:
    enabled: true
    window: 50          # baseline size
    z_threshold: 3.0    # sensitivity (lower = more alerts)
    min_baseline: 5     # history required before alerts fire
  review:
    require_approval: true
    default_export_statuses: ["approved"]
```

Set `ATLAS_DRIFT_WINDOW`, `ATLAS_DRIFT_Z_THRESHOLD`, or `ATLAS_DRIFT_MIN_BASELINE` to experiment without editing config. `ATLAS_REVIEW_DEFAULT_EXPORT_STATUSES="approved,pending"` relaxes the default export gate for developer workflows, and `ATLAS_REVIEW_REQUIRE_APPROVAL=0` disables the auto-approval requirement entirely (use with care in production).
