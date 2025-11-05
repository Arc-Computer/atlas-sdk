# Configuration Guide

Atlas orchestrates three collaborating systems—the **student agent**, the **teacher coach**, and the **reward system** that measures progress. This guide gives you practical levers for steering each component so you can align Atlas with your domain, latency targets, and cost envelope.

> **Reference:** Facts in this document come directly from the source code. Default values are taken from `atlas/config/models.py` and the runtime configurators under `atlas/config`.

## Key Concepts

Before diving into configuration, here's how Atlas works:

- **Student agent** executes your task using an LLM. You control its model, prompt, and adapter type.
- **Teacher** reviews student plans and can provide step-by-step guidance when needed. A **capability probe** (small LLM) decides when teacher intervention is necessary based on confidence scores.
- **Reward system** scores student outputs using judge models. Scores feed into learning.
- **Learning synthesizer** distills successful patterns into **playbook entries** (e.g., "when reviewing auth code, check for server-side validation"). These entries are injected back into prompts when matching patterns are detected.

**Empirical validation:** Atlas uses outcome-based validation—entries are accepted provisionally and validated based on real-world performance metrics (cue hit rate, reward improvement, transfer success) rather than syntactic rules. This allows domain-specific terminology while filtering out ineffective patterns.

## Student Agent

The agent adapter defines how Atlas invokes your runtime:

```yaml
agent:
  type: litellm                # also accepts python or http_api
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

- **LiteLLM adapter** (see `atlas/config/models.py::LitellmAdapterConfig`) works with multiple providers (OpenAI, Anthropic, Gemini, Bedrock, X.AI, Azure OpenAI) via litellm while exposing response-formatting and metadata-digest controls.
- **Python adapter** wraps a local async or sync function. The SDK passes prompt metadata (including optional `llm_config`) so your code can call downstream models directly.
- **HTTP adapter** templates arbitrary REST endpoints and extracts responses with JSONPath sequences.

**Metadata digestion:** Atlas automatically summarizes session history and learning context to fit within model context windows. This reserves ~10% of a model's published context window and can be tightened via `metadata_digest.char_budget`, `max_learning_history_entries`, or `include_session_keys`.

## Teacher Coach

The teacher operates in three modes, selected by a **capability probe**—a small LLM that assesses task difficulty and confidence before execution:

```yaml
teacher:
  llm:
    provider: openai
    model: gpt-4.1
    api_key_env: OPENAI_API_KEY
    temperature: 0.05
```

The capability probe runs before each task and selects a mode based on confidence scores (see Adaptive Teaching section for configuration):

- **Paired mode**: teacher audits the plan before execution.
- **Coach mode**: teacher guides each intermediate step when confidence dips.
- **Auto mode**: no teacher involvement—student runs alone.

Reducing thresholds (see the adaptive teaching section) expands auto mode and cuts cost. Raising them increases review coverage.

## Reward System

The block defines a two-tier judge stack that scores every completed task and feeds learning:

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

The learning system distills successful patterns into **playbook entries**—reusable guidance that gets injected into prompts when matching patterns are detected. For example, a playbook entry might be: "when reviewing authentication code, check for server-side validation."

**How it works:**
1. After each session, the synthesizer reviews recent runs and identifies patterns
2. Patterns are converted into playbook entries with triggers (cues) and guidance
3. Entries are accepted **provisionally** and validated based on real-world performance
4. Entries that prove valuable (high cue hit rate, reward improvement, transfer success) remain active
5. Entries that don't perform well are **pruned** automatically

**Empirical validation:** Entries are validated based on measurable outcomes rather than syntactic rules. This allows domain-specific terminology (e.g., "JWT", "IDOR") while filtering out truly ineffective patterns.

### Configuration

Default values are defined in code (`LearningConfig.llm` sets Gemini 2.5 Flash with temperature 0.1 and 8,192 max tokens). For most use cases, you only need to configure the LLM and gates:

```yaml
learning:
  llm:
    provider: gemini
    model: gemini/gemini-2.5-flash
    api_key_env: GEMINI_API_KEY
    temperature: 0.1
    max_output_tokens: 8192
    timeout_seconds: 120
  gates:
    enforce_actionability: true
    enforce_cue: true
```

**Default Behavior (no configuration needed):**

- **enabled**: `true` - Learning is enabled by default
- **update_enabled**: `true` - Learning updates after each session
- **provisional_acceptance**: `true` - Entries are accepted initially and validated later based on metrics
- **empirical validation**: Enabled by default. Entries are validated based on performance (cue hit rate, reward delta, transfer success) rather than syntactic rules
- **pruning_config**: Automatic pruning with sensible defaults (min_sessions: 10, min_cue_hit_rate: 0.05, etc.)

**Only specify what you need to customize.** For most use cases, you only need to configure the LLM and gates.

**Common Tuning Options:**

These options are frequently adjusted but have sensible defaults:

```yaml
learning:
  # ... llm and gates ...
  
  # Learning update frequency
  history_limit: 10          # How many recent runs to review (default: 10)
                             # Lower = faster convergence, Higher = deeper patterns
  
  # Learning injection control
  apply_to_prompts: true     # Inject learning into prompts (default: true)
                             # Set false to capture learning without affecting live traffic
  
  # Empirical validation tuning
  pruning_config:
    min_sessions: 10         # Minimum sessions before pruning (default: 10)
    min_cue_hit_rate: 0.05   # Minimum cue hit rate to avoid "too specific" pruning (default: 0.05)
                             # Cue = trigger pattern, hit rate = how often it fires
    min_reward_delta: 0.01   # Minimum reward improvement (default: 0.01)
    min_transfer_sessions: 20 # Sessions needed for transfer check (default: 20)
                             # Transfer = entry works across multiple incidents/tasks
```

**When to tune:**

- **`history_limit`**: Lower (5-8) for faster iteration, higher (15-20) for broader patterns
- **`apply_to_prompts`**: Set `false` when measuring baseline or capturing without injection
- **`pruning_config`**: Adjust thresholds if entries are pruned too aggressively or not enough
  - **`min_cue_hit_rate`**: Higher (0.1-0.2) = keep only frequently-used patterns, lower (0.01-0.03) = allow niche patterns
  - **`min_reward_delta`**: Higher (0.02-0.05) = keep only entries with clear impact, lower (0.001-0.005) = allow marginal improvements
  - **`min_sessions`**: Lower (5-8) = prune sooner with less data, higher (15-20) = wait for more statistical confidence

**Real-World Tuning Examples:**

1. **Faster iteration during development:**
   ```yaml
   learning:
     history_limit: 5  # Review fewer sessions for faster updates
     pruning_config:
       min_sessions: 5  # Prune sooner with less data
   ```

2. **Stricter quality control:**
   ```yaml
   learning:
     pruning_config:
       min_cue_hit_rate: 0.15  # Keep only frequently-used patterns
       min_reward_delta: 0.03  # Require clear impact (3% improvement)
   ```

3. **Capture learning without affecting production:**
   ```yaml
   learning:
     apply_to_prompts: false  # Learn but don't inject into prompts
   ```

4. **Tool-specific constraints:**
   ```yaml
   learning:
     schema:
       allowed_runtime_handles: [read_file, search_content, run_command]
       cue_types: [regex]  # Only regex cues
   ```

**Schema Constraints:**

If you need to restrict which tools or cue types are allowed:

```yaml
learning:
  # ... llm and gates ...
  schema:
    allowed_runtime_handles: [read_file, search_content]  # Specific tools allowed
    runtime_handle_prefixes: [logs.*, data.*]  # Prefix patterns
    cue_types: [regex, keyword]  # Limit cue types (default: all)
    default_scope_category: reinforcement  # reinforcement | differentiation
```

**Advanced Configuration:**

If you need to customize empirical validation behavior:

```yaml
learning:
  gates:
    enforce_generality: false  # Default: use empirical validation
                               # Set to true for strict immediate rejection (legacy behavior)
```

- **enforce_generality** (default: `false`) - Controls how generality gate failures are handled:
  - `false` (recommended): Gate runs but failures are non-blocking. Entries are accepted provisionally and pruned later based on empirical metrics (cue hit rate, reward delta, transfer success).
  - `true` (legacy): Gate failures cause immediate rejection. No empirical validation - uses predictive syntactic gates only. May reject valuable entries that contain domain terminology or are longer than expected.

The generality gate checks for obvious anti-patterns (dates, incident IDs, file paths, banned tokens) regardless of this setting. The difference is whether failures block acceptance or allow provisional acceptance.

### Additional Details

- **Prompt format**: Playbook entries are injected into prompts with delimiters (`>>> Student Playbook >>>`, `>>> Teacher Playbook >>>`) and trimmed to roughly 1,000 characters. Metadata such as version or timestamp is surfaced on a single header line when present.
- **Caching behavior**: The teacher's plan cache includes a digest of injected playbook entries, so changing learning content invalidates stale reviews without flushing the entire cache.
- **Schema + gates**: Enforce tool fidelity and prevent overfit entries. Entries referencing handles outside `allowed_runtime_handles` are rejected at validation time. See the "Schema Constraints" section above for configuration examples.
- **Evaluation harnesses**: When running learning or runtime evaluation harnesses, keep `apply_to_prompts` enabled so captured metrics reflect the adaptive system end-to-end. Disable it only after you have baseline numbers for comparison.

## Adaptive Teaching

The **capability probe** is a small LLM that runs before each task to assess difficulty and confidence. Based on its confidence score, it selects one of three teacher modes:

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

- **Confidence ≥ auto threshold** → **auto** mode (student runs alone)
- **Between auto and paired** → **paired** mode (teacher reviews plan before execution)
- **Between paired and coach** → **coach** mode (teacher guides each step)
- **Below coach threshold** → fall back to `fallback_mode` (default: **paired**)

**Why this matters:** The probe runs on every task, so choose an inexpensive, fast model. Latency adds directly to user-facing wait times.

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

Monitor the impact of configuration changes directly from the `sessions` table in PostgreSQL:

- **Average duration (seconds)** - Measures end-to-end latency
  ```sql
  SELECT
    AVG(EXTRACT(EPOCH FROM (completed_at - created_at))) AS avg_duration_seconds
  FROM sessions
  WHERE completed_at IS NOT NULL
    AND created_at > NOW() - INTERVAL '7 days';
  ```
- **Average token usage** - Tracks cost per session
  ```sql
  SELECT
    AVG((metadata->'token_usage'->>'total_tokens')::numeric) AS avg_total_tokens
  FROM sessions
  WHERE metadata ? 'token_usage'
    AND created_at > NOW() - INTERVAL '7 days';
  ```
- **Learning adoption** - Percentage of sessions where playbook entries were applied
  ```sql
  SELECT
    COUNT(*) FILTER (WHERE metadata->>'applied_student_learning' IS NOT NULL) * 100.0 / COUNT(*) AS adoption_pct
  FROM sessions
  WHERE created_at > NOW() - INTERVAL '7 days';
  ```

**Interpreting results:**
- **Low adoption** (<20%): Cues are too narrow or entries are being rejected. Consider lowering `min_cue_hit_rate` or checking gate configurations.
- **High adoption without measurable uplift** (>70% but no reward improvement): Reward prompt may not be emphasizing the right behaviors, or entries aren't actually helpful.
- **Good adoption with uplift** (30-70% with reward improvement): System is learning effectively.

## Next Steps

- Start from `configs/examples/openai_agent.yaml` and tailor the agent/teacher section.
- Keep reward-system prompts in version control; small prompt tweaks have large scoring effects.
- Re-run the evaluation harnesses in `docs/evaluation/` whenever you change providers or prompts so the student, teacher, and reward system stay calibrated together.
