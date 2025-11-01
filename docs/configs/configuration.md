# Configuration Guide

Atlas orchestrates three collaborating systems—the **student agent**, the **teacher coach**, and the **reward system** that measures progress. This guide gives you practical levers for steering each component so you can align Atlas with your domain, latency targets, and cost envelope.

> Reference facts in this document come directly from the source code. Default values are taken from `atlas/config/models.py` and the runtime configurators under `atlas/config`.

## Architecture Overview

- **Student agent** executes the user task. You control its adapter (OpenAI-compatible, HTTP, or Python callable) and prompt surface.
- **Teacher** reviews plans and can coach step-by-step when the capability probe signals elevated risk.
- **Reward system** scores the student output asynchronously using one or more judge models. Scores and rationales flow into learning.
- **Learning synthesizer** (fed by the reward system) distills durable playbook entries and injects them back into student/teacher prompts when activations match.

Atlas keeps the systems loosely coupled: swap any model stack without touching orchestration logic, and tune thresholds to decide when each role wakes up.

## Student Agent

The agent adapter defines how Atlas invokes your runtime:

```yaml
agent:
  type: openai                # also accepts python or http_api
  name: security-analyst
  system_prompt: |
    You are the Atlas student. Produce actionable, evidence-backed responses.
  llm:
    provider: openai
    model: gpt-4.1-mini
    api_key_env: OPENAI_API_KEY
    temperature: 0.1
    max_output_tokens: 8192
    timeout_seconds: 180
```

- **OpenAI adapter** (see `atlas/config/models.py::OpenAIAdapterConfig`) works with any OpenAI-compatible endpoint while exposing response-formatting and metadata-digest controls.
- **Python adapter** wraps a local async or sync function. The SDK passes prompt metadata (including optional `llm_config`) so your code can call downstream models directly.
- **HTTP adapter** templates arbitrary REST endpoints and extracts responses with JSONPath sequences.

Metadata digestion is enabled by default. The digest budget reserves ~10% of a model’s published context window and can be tightened via `metadata_digest.char_budget`, `max_learning_history_entries`, or `include_session_keys`.

## Teacher Coach

Teachers operate in three modes driven by the capability probe:

```yaml
teacher:
  llm:
    provider: openai
    model: gpt-4.1
    api_key_env: OPENAI_API_KEY
    temperature: 0.05
```

- **Paired mode**: teacher audits the plan before execution.
- **Coach mode**: teacher guides each intermediate step when confidence dips.
- **Auto mode**: no teacher involvement.

Reducing thresholds (see the adaptive teaching section) expands auto mode and cuts cost. Raising them increases review coverage.

## Reward System

The reward system replaces autonomous “RIM” terminology from earlier docs. Configuration still uses the `rim` key for backward compatibility; the behaviour is unchanged. The block defines a two-tier judge stack that scores every completed task and feeds learning:

```yaml
rim:
  small_model:
    provider: gemini
    model: gemini/gemini-2.5-flash
    api_key_env: GEMINI_API_KEY
    max_output_tokens: 8192
  large_model:
    provider: gemini
    model: gemini/gemini-2.5-pro
    api_key_env: GEMINI_API_KEY
    max_output_tokens: 8192
  variance_threshold: 0.15
  uncertainty_threshold: 0.30
  judge_prompt: |
    Reward grounded, verifiable answers that follow safety guardrails.
```

1. **Small model pass** (Gemini 2.5 Flash by default) runs every time.
2. **Large model escalation** (Gemini 2.5 Pro in the example) triggers when judge variance or self-reported uncertainty breach configured thresholds.

Tune the thresholds to trade cost for consistency. Increasing `variance_threshold` toward 0.30 reduces escalations; tightening it to ~0.10 yields more second-opinion arbitrations.

## Learning Synthesizer

Default values for the learning synthesizer now live in code (`LearningConfig.llm` sets Gemini 2.5 Flash with temperature 0.1 and 8 192 max tokens). Configuration controls both the synthesis schedule and the structure of the emitted playbook entries:

```yaml
learning:
  enabled: true
  update_enabled: true
  llm:
    provider: gemini
    model: gemini/gemini-2.5-flash
    temperature: 0.1
    max_output_tokens: 8192
    timeout_seconds: 120
  history_limit: 10
  apply_to_prompts: true
  schema:
    allowed_runtime_handles:
      - read_file
      - search_content
    cue_types: [regex, keyword, predicate]
  gates:
    enforce_actionability: true
    enforce_cue: true
    enforce_generality: true
    max_text_length: 420
  rubric_weights:
    actionability: 0.4
    generality: 0.3
    hookability: 0.2
    concision: 0.1
```

- **history_limit** decides how many recent runs the synthesizer reviews at once. Lower values trigger quicker feedback loops; higher values capture longer-term patterns.
- **apply_to_prompts** toggles prompt injection. Set it to `false` to capture learning while keeping live traffic untouched.
- **apply_to_prompts (runtime override)** defaults to `true`, so both student and teacher prompts include pam­phlets when the synthesizer has entries for the active learning key. Flip it to `false` when you want to measure a baseline without adaptive context.
- **Prompt format**: injected blocks are wrapped with lightweight delimiters (`>>> Student Playbook >>>`, `>>> Teacher Playbook >>>`) and trimmed to roughly 1 000 characters. Metadata such as version or timestamp is surfaced on a single header line when present in `learning_state["metadata"]`.
- **Caching behaviour**: the teacher’s plan cache includes a digest of the injected pamphlet, so changing learning content invalidates stale reviews without flushing the entire cache.
- **schema + gates** enforce tool fidelity and prevent overfit entries. Entries referencing handles outside `allowed_runtime_handles` are rejected at validation time.
- **Evaluation harnesses**: when running the learning or runtime evaluation harnesses, keep `apply_to_prompts` enabled so the captured metrics reflect the adaptive system end to end. Disable it only after you have baseline numbers for comparison.

## Adaptive Teaching

Capability probes determine when the teacher intervenes. The probe runs before each task and selects a mode based on confidence scores.

```yaml
adaptive_teaching:
  enabled: true
  probe:
    llm:
      provider: xai
      model: xai/grok-2-mini
      api_key_env: XAI_API_KEY
    thresholds:
      auto: 0.85
      paired: 0.65
      coach: 0.35
    fallback_mode: paired
  learning_history_limit: 10
```

- Confidence ≥ auto threshold → **auto** (student runs alone).
- Between auto and paired → **paired** (teacher reviews plan).
- Between paired and coach → **coach** (teacher in the loop).
- Below coach threshold → escalate to human or fall back to `fallback_mode`.

Choose a probe model that is inexpensive and responsive—the probe runs on every task, so latency adds directly to user-facing wait times.

## Storage & Persistence

Persistent storage unlocks telemetry, learning, and reporting:

```yaml
storage:
  database_url: postgresql://atlas:atlas@localhost:5433/atlas
  min_connections: 1
  max_connections: 5
  statement_timeout_seconds: 30
```

Atlas expects PostgreSQL. Set the connection pool based on concurrency, and keep `statement_timeout_seconds` low enough to prevent wedged queries.

## Configuration Playbooks

Use these starting points and adjust to your workload.

### Cost-Optimised Stack

- Student: `gpt-4.1-mini`
- Teacher: `gpt-4.1-mini`
- Reward system: Gemini 2.5 Flash for both tiers with `variance_threshold: 0.25`
- Learning: keep defaults, optionally drop `history_limit` to 6 for faster convergence
- Adaptive teaching: `enabled: false` for fully autonomous runs

This profile minimises judge escalations and removes probe/teacher overhead while preserving learning.

### Quality-First Stack

- Student: `gpt-4.1`
- Teacher: `gpt-4o`
- Reward system: Gemini 2.5 Flash (small) + Gemini 2.5 Pro (large), `variance_threshold: 0.10`
- Learning: raise `gates.max_text_length` to 800 for richer guidance
- Adaptive teaching: tighten thresholds (`auto: 0.92`, `paired: 0.75`, `coach: 0.45`)

Choose this preset when correctness outweighs latency or cost.

### Latency-Sensitive Stack

- Student: `gpt-4.1-nano`
- Teacher: disabled (`adaptive_teaching.enabled: false`)
- Reward system: 2× Gemini 2.5 Flash with `variance_threshold: 0.30`
- Learning: set `update_enabled: false` to avoid synthesis during live traffic
- Storage: keep enabled for auditability even when learning is paused

This keeps the critical path to a single fast model call while still collecting reward signals for offline review.

## Observability & Feedback Loops

Monitor the impact of configuration changes directly from the `sessions` table:

- **Average duration (seconds)**
  ```sql
  SELECT
    AVG(EXTRACT(EPOCH FROM (completed_at - created_at))) AS avg_duration_seconds
  FROM sessions
  WHERE completed_at IS NOT NULL
    AND created_at > NOW() - INTERVAL '7 days';
  ```
- **Average token usage**
  ```sql
  SELECT
    AVG((metadata->'token_usage'->>'total_tokens')::numeric) AS avg_total_tokens
  FROM sessions
  WHERE metadata ? 'token_usage'
    AND created_at > NOW() - INTERVAL '7 days';
  ```
- **Learning adoption**
  ```sql
  SELECT
    COUNT(*) FILTER (WHERE metadata->>'applied_student_learning' IS NOT NULL) * 100.0 / COUNT(*) AS adoption_pct
  FROM sessions
  WHERE created_at > NOW() - INTERVAL '7 days';
  ```

Low adoption means cues are too narrow or entries are being rejected. High adoption without measurable uplift generally points to a reward prompt that is not emphasising the right behaviours.

## Next Steps

- Start from `configs/examples/openai_agent.yaml` and tailor the agent/teacher section.
- Keep reward-system prompts in version control; small prompt tweaks have large scoring effects.
- Re-run the evaluation harnesses in `docs/evaluation/` whenever you change providers or prompts so the student, teacher, and reward system stay calibrated together.
