# Learning Evaluation Without Experience Hints

Atlas already captures the signals needed to explain what changed, how it changed, and why—even before hint
distillation ships. This guide documents the end-to-end workflow for analysing learning progress using the telemetry
persisted by the runtime today.

> **Terminology update (2025-10-29):** Former "policy nugget" references have been renamed to **playbook entries**. Regenerate any stored telemetry created before 2025-10-29 to align with the new schema.

## Prompt Digest For Provider Limits

Learning evaluations now route execution metadata through a **provider-aware prompt digest** before the adapter sends
requests to the LLM. This prevents 200k+ token system messages from blocking Claude and other providers with
smaller context windows while keeping the full telemetry available on disk.

- The OpenAI-compatible adapter exposes a new `metadata_digest` block. Defaults trim large sections (reward audits,
  session trajectories, validation blobs) to a high-signal summary capped at roughly 10% of each provider's context
  window (≈20k characters for Anthropic).
- Each digest produced for an LLM includes `digest_stats` (budget used, omitted metadata keys, and any sections
  dropped to stay under budget). These diagnostics live in the system message payload for troubleshooting.
- Override defaults per workflow:

```yaml
agent:
  adapter:
    type: openai
    metadata_digest:
      char_budget: 24000        # Optional hard cap for every provider
      provider_char_budgets:
        anthropic: 18000        # Override Claude/Sonnet to stay safely below 200k tokens
      max_plan_steps: 6         # Control how many plan steps appear in the digest
      max_learning_history_entries: 2
      include_session_keys: [source, execution_mode, token_usage, reward_stats]
```

- Set `enabled: false` to revert to the legacy behaviour (not recommended for Claude/Bison-sized windows).
- If the digest cannot fit under the configured budget after trimming optional sections it raises a descriptive
  error instead of attempting to send the oversized payload.

Gemini continues to receive the same or smaller prompts, while Anthropic and other providers now stay well within
their context limits during benchmarking runs.

## Playbook Entry Schema & Rubric

Learning updates now revolve around structured **playbook entries**. Each playbook entry captures:

- **cue** – regex/keyword trigger that can be machine-detected.
- **action** – imperative phrasing plus the runtime handle/tool mapping.
- **expected_effect** – why the action matters.
- **scope** – whether the playbook entry reinforces an existing behaviour or introduces differentiation, including any constraints.
- **provenance** – session id, teacher intervention digest, rubric scores, and lifecycle (`active`, `deprecated`, `rejected`).

Three rubric gates run on every synthesis:

1. **Actionability** – the handle must map to a real tool and the imperative cannot be empty.
2. **Cue presence** – cues must be machine-detectable (valid regex/keyword/predicate).
3. **Generality** – no incident IDs/dates or overfit proper nouns; playbook entries must respect a length budget.

Scores for actionability, generality, hookability, and concision (weights: 0.4 / 0.3 / 0.2 / 0.1) are computed even when gates fail. If any gate fails the existing pamphlet is preserved and the rejection is recorded for auditing.

### Configuring schema, gates, and instrumentation

Atlas reads these rails from the existing `learning` block in your agent config (for example `configs/<project>.yaml`). If you omit the block, Atlas instantiates the default `LearningConfig`. To enable stricter constraints or adjust weights, add a section like:

```yaml
learning:
  enabled: true
  update_enabled: true
  schema:
    allowed_runtime_handles:
      - logs.search
      - data.query*
    cue_types: [regex, keyword]
    default_scope_category: reinforcement
  gates:
    enforce_actionability: true
    enforce_cue: true
    enforce_generality: true
    max_text_length: 420
    allowed_proper_nouns: [SQL, HTTP, JSON, Atlas]
  rubric_weights:
    actionability: 0.4
    generality: 0.3
    hookability: 0.2
    concision: 0.1
  usage_tracking:
    enabled: true
    capture_examples: true
    max_examples_per_entry: 3
```

- `schema` constrains what the LLM can emit (permitted runtime handles/prefixes, cue types, default scope category).
- `gates` toggles the rubric guards and tunes generalisation heuristics (length budget, banned tokens, allowlists).
- `rubric_weights` rebias the weighted playbook entry score if you want concision or hookability to matter more/less.
- `usage_tracking` enables cue/adoption logging and limits how many example snippets are stored per playbook entry.

All other `learning` options (`llm`, `prompts`, `history_limit`, `session_note_enabled`, `apply_to_prompts`) behave as before. Once configured, every synthesis run honours these settings automatically.

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

## Impact Metrics (Adaptive Efficiency & Transfer)

Runtime instrumentation now enriches every session with an `impact` snapshot so we can reason about how each
playbook entry contributes to adaptive efficiency (faster wins on known tasks) and cross-incident transfer (reusing
guidance when the incident changes). The tracker captures reward/token deltas, incident identifiers, retry counts,
and failure summaries per entry. The evaluation harness aggregates these into the playbook metadata under
`playbook_entries[].impact` and exposes a dedicated **Playbook Entry Impact** section in both JSON and Markdown.

- **Adoption rate** – successful adoptions ÷ cue hits. A hit without adoption indicates guidance being seen but not
  followed; sustained adoption >60 % is a good reinforcement signal.
- **Reward delta** – average reward for sessions where the entry fired minus the average reward when it did not.
  Positive deltas demonstrate adaptive efficiency (more wins when guidance triggers); negative deltas suggest
  the entry may be stale or misleading.
- **Token delta** – average tokens with the entry firing minus tokens without it. Negative numbers imply efficiency
  gains (doing the job in fewer tokens); positive spikes highlight regressions in runtime cost.
- **Transfer success** – marked true when the entry triggers across at least two distinct incident/task identifiers.
  This is the lightweight proxy for cross-incident reuse described in the *Continual Learning Online Adaptation* memo.
- **Failure avoidance stats** – rolling average retries and recorded failure events when the entry fires. Falling retry
  counts or zero failure events indicate the entry is preventing repeat mistakes.
- **Impact score** – `adoption_rate × reward_delta`. This composite favors entries that are both frequently adopted and
  deliver positive reward deltas. Treat it as a prioritisation heuristic when curating the playbook: entries with
  negative scores should be audited first.

The same signals are stored session-by-session under `metadata.learning_usage.session` so you can audit individual
runs or recompute experiment-specific aggregates.

## 3. Run the Learning Evaluation Script

Use the new `scripts/eval_learning.py` helper to assemble structured summaries per learning key. The script queries
Postgres directly—no JSONL export required—and produces JSON + Markdown reports under `results/learning/`.

```bash
python scripts/eval_learning.py \
  --database-url postgresql://atlas:atlas@localhost:5433/atlas \
  --recent-window 10 \
  --baseline-window 50 \
  --limit 5 \
  --prompt-variant schema_v2 \
  --synthesis-model gpt-4o-mini --synthesis-model claude-3-sonnet \
  --pamphlet-injection toggle
```

> **Pamphlet verification**: leave `learning.apply_to_prompts` at its default
> (`true`) when running the script so the generated summaries reflect the same
> guidance injected into runtime prompts. The resulting
> `results/learning/*_summary.json` entries include the most recent
> `teacher_playbook` payload when the pamphlet is present, making it easy to
> confirm the runtime wiring without hitting the live APIs.

### Key options

- `--summary-only` – skips per-session trajectory fetches and relies on SQL event counts; use this for large sweeps or CI.
- `--batch-size` – max number of learning keys evaluated concurrently (default: 4).
- `--filter-project`, `--filter-task`, `--filter-tag` – narrow the run to a specific codebase, task name, or session tags.
- `--learning-key` – analyze explicit keys instead of querying Postgres for the top-N recent keys.
- `--compare-to results/learning/index.json` – diff the current run against a previous harness export; the manifest stores per-key deltas and Markdown files append a comparison section.
- `--no-markdown` – emit only machine-readable JSON for automation scenarios.
- `--prompt-variant` – label the prompt/meta-prompt variant under test.
- `--synthesis-model` – record the LLM(s) used for pamphlet generation (repeatable, feeds model benchmarking comparisons).
- `--pamphlet-injection` – annotate whether pamphlet injection was on/off/toggled for transfer tests.
- `--playbook-entry-labels` – reference a JSON file with manual playbook entry category overrides (stored in the manifest for downstream tooling).

Summary mode is ideal for nightly or CI jobs where you just need reward deltas and model trends. Run the full-detail mode (default) when you want trajectory event counts sampled per session and are comfortable with additional database reads.

Outputs:

- `results/learning/<slug>_summary.json` – machine-readable payload (sessions, reward windows, discovery references).
- `results/learning/<slug>_summary.md` – human-friendly digest highlighting reward deltas, adaptive behaviour, and model breakdowns.
- `results/learning/index.json` – manifest listing every generated artifact, plus the comparison/aggregate tables when `--compare-to` is provided.
- `playbook_impact` (in both JSON + Markdown summaries) – per-entry adoption, reward/token delta, transfer, failure avoidance, and composite `impact_score` metrics for adaptive-efficiency tracking.
- `run_metadata` (in `index.json`) – captures prompt variant, synthesis models, pamphlet toggle mode, and optional playbook entry label overrides supplied via CLI flags.

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
- **Playbook Entry Quality** – aggregates the rubric outputs: candidate counts, gate failures, weighted score averages, and the weighting used.
- **Playbook Entry Lifecycle** – reinforcement vs differentiation counts split by `active`/`deprecated`, plus rejected candidates from the latest run.
- **Playbook Entry Impact** – adoption rate, reward/token deltas, transfer success, failure avoidance signals, and the composite `impact_score` for each entry so you can prioritise curation according to adaptive-efficiency gains.
- **Runtime Usage** – cue trigger totals, adoption counts, success rates, and trigger/adoption rates across sessions.
- **Efficiency Snapshot** – comparison of reward/tokens in sessions with cue hits versus those without, including deltas.

Because everything keys off `learning_key`, you can join the summary back to:

- `sessions` (runtime telemetry + reward signals)
- `discovery_runs` (discovery/runtime captures recorded via the CLI)
- `learning_registry` (current pamphlet state, when enabled)

## 5. Compare Runs Over Time

When you pass `--compare-to`, the harness looks up the previous `index.json`, loads each saved summary, and computes deltas for:

- Reward trends (recent mean, latest score)
- Session counts per learning key
- Model-level utilisation and reward mean changes
- Cue hit/adoption changes (derived from the usage metrics section)

The new manifest includes `comparisons` and aggregate leaderboards (best/worst deltas), while each Markdown report gains a “Comparison vs previous run” section.

## 6. Suggested Experiments

To stress-test the learning synthesizer and meta-prompt variants, run targeted sweeps with the configs under
`configs/eval/`:

- `learning_overhaul_base.yaml` — baseline Gemini 2.5 Flash synthesiser and reinforcement-focused prompt.
- `learning_overhaul_scope_shift.yaml` — emphasises differentiation and transfer hypotheses; default scope category set to `differentiation`.
- `learning_overhaul_claude.yaml` — Claude Haiku/Sonnet stack for student/teacher/synthesiser evaluation.

Generate fresh telemetry for each config (same dataset, different `learning_key`s), then compare `playbook_impact`
sections across runs. Prioritise variants that increase adoption rate without regressing token deltas, and flag any
entries with negative impact scores for remediation.

## 6. Automate & Test

- The evaluation script ships with unit tests that stub database access and external LLM calls, so `pytest` covers the
  new entry points without touching live services.
- To keep the workflow reproducible, commit the generated summaries or re-run the script as part of your evaluation
  pipeline once fresh telemetry lands.
- When experimenting with prompt variants or different synthesis models, record the configuration via the new CLI flags so the manifest preserves the experimental context.

With these pieces in place we can meaningfully answer “what changed, how it changed, and why” today, deferring the
hint-specific analytics until the hint pipeline arrives.
