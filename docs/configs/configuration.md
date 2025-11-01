# Configuration Guide

Atlas implements adaptive teaching through three models: the agent (student), teacher, and RIM (reward integrity model). Configuration controls how these components interact and when interventions occur.

## Core Architecture

The agent executes tasks while the teacher provides validation and guidance based on capability assessment. RIM evaluates execution quality through multi-judge consensus, triggering learning synthesis when patterns emerge. Learning manifests as playbook entries that inject into student prompts when activation cues match.

All three models can use different providers and tiers. The most common pattern uses a strong agent model, reuses it for teacher validation, and delegates reward evaluation to cheaper models since RIM runs after every task.

## Agent Configuration

The agent adapter determines how Atlas invokes your model. OpenAI adapters work with any OpenAI-compatible endpoint and support structured outputs via `response_format`. Python adapters wrap custom implementations that return strings or generators. HTTP adapters proxy arbitrary endpoints by templating payloads and extracting results via JSONPath.

```yaml
agent:
  type: openai
  name: my-agent
  system_prompt: "Task-specific instructions"
  llm:
    provider: openai
    model: gpt-4.1-mini
    api_key_env: OPENAI_API_KEY
  response_format:
    type: json_schema
    json_schema: {...}
```

OpenAI adapters inject execution metadata into prompts through `metadata_digest` controls. The digest budget defaults to 10% of the model's context window and includes recent step summaries, learning history, reward statistics, and drift alerts. Increasing `char_budget` surfaces more execution context at the cost of prompt tokens. The `include_session_keys` list filters which metadata categories appear. Reducing `max_learning_history_entries` limits how many past learning cycles the agent sees, trading historical context for token efficiency.

## Teacher Configuration

The teacher model validates plans in paired mode and provides step-by-step guidance in coach mode. It never activates in auto mode. Teacher LLM selection trades cost against validation quality. Using the same model as the agent maintains consistency but doubles inference cost during paired execution. Using a weaker model reduces cost but may produce lower-quality guidance.

```yaml
teacher:
  llm:
    provider: openai
    model: gpt-4.1-mini
```

Teacher invocation frequency depends on adaptive teaching thresholds. Lowering the auto threshold increases autonomous execution, reducing teacher calls. Raising the paired threshold pushes more tasks into coach mode, increasing teacher involvement. The probe evaluates agent capability per-task and selects the execution mode accordingly.

## RIM Configuration

RIM performs two-stage evaluation using small and large models. The small model handles initial judgment across multiple aspects in parallel. When judge variance exceeds `variance_threshold` or uncertainty exceeds `uncertainty_threshold`, the large model arbitrates. This architecture balances cost and consistency.

```yaml
rim:
  small_model:
    provider: gemini
    model: gemini/gemini-2.5-flash
    max_output_tokens: 8192
  large_model:
    provider: gemini
    model: gemini/gemini-2.5-flash
    max_output_tokens: 8192
  variance_threshold: 0.15
  uncertainty_threshold: 0.3
  judge_prompt: |
    Domain-specific evaluation criteria
```

The small model runs 2-3 parallel judgments per task. Large model escalation occurs in 20-40% of evaluations at default thresholds. Raising `variance_threshold` to 0.25-0.30 reduces escalations, cutting costs but potentially introducing scoring inconsistency. The `judge_prompt` steers evaluation criteria toward domain objectives. Clear, specific prompts reduce variance and improve score reliability.

Using identical models for small and large tiers works when the model has sufficient capability for judgment. Using a stronger large model improves tie-breaking quality when small model judgments conflict. The cost difference only matters during escalation, so this optimization has bounded impact.

## Learning Configuration

Learning synthesizes playbook entries from execution traces and injects them into agent prompts when cues activate. The system defaults to Gemini Flash for synthesis since this operation runs asynchronously after task completion. Synthesis frequency depends on `history_limit`, which controls how many executions accumulate before triggering analysis.

```yaml
learning:
  enabled: true
  update_enabled: true
  llm:
    provider: gemini
    model: gemini/gemini-2.5-flash
    temperature: 0.1
    max_output_tokens: 8192
  schema:
    allowed_runtime_handles:
      - read_file
      - write_file
    cue_types: [regex, keyword, predicate]
  gates:
    enforce_actionability: true
    enforce_cue: true
    enforce_generality: true
    max_text_length: 800
  rubric_weights:
    actionability: 0.4
    generality: 0.3
    hookability: 0.2
    concision: 0.1
  history_limit: 10
  apply_to_prompts: true
```

The schema section defines which runtime handles (tool names) can appear in synthesized entries. Entries referencing unlisted handles fail validation. This prevents hallucinated tool references when the LLM generates playbook guidance. The `cue_types` list enables different activation patterns: keyword matching for exact phrases, regex for flexible patterns, and predicate for structured conditions on execution context.

Gates enforce quality constraints. Actionability requires entries to specify concrete tool usage rather than abstract advice. Cue enforcement mandates clear activation triggers so entries don't fire indiscriminately. Generality prevents overfitting to specific task instances. The `max_text_length` limit caps entry size, directly controlling prompt token overhead when entries inject. Default 420 characters balances detail against efficiency. Increasing to 600-800 allows richer guidance but inflates prompts.

Rubric weights determine how entries score during validation. Higher actionability weight prioritizes concrete tool patterns over general principles. Higher generality weight favors transferable lessons over task-specific solutions. Adjusting these weights steers what kinds of knowledge the system retains.

Setting `apply_to_prompts: false` disables prompt injection while still generating and storing playbook entries. This mode supports analysis and export workflows without affecting runtime behavior.

## Adaptive Teaching

Adaptive teaching controls when teacher intervention occurs based on capability probes. The probe runs before each task, evaluating whether the agent can handle it autonomously.

```yaml
adaptive_teaching:
  enabled: true
  probe:
    llm:
      provider: xai
      model: xai/grok-4-fast
    thresholds:
      auto: 0.85
      paired: 0.65
      coach: 0.35
    fallback_mode: paired
  learning_history_limit: 10
```

Probe thresholds define mode transitions. Confidence above 0.85 triggers auto mode (no teacher). Confidence 0.65-0.85 triggers paired mode (teacher validates plan). Confidence 0.35-0.65 triggers coach mode (teacher guides each step). Below 0.35 escalates to human review.

Lowering the auto threshold increases autonomous execution, reducing teacher overhead. Raising it pushes more tasks into paired mode for validation. The paired threshold controls the auto-to-paired boundary, while coach threshold controls paired-to-coach escalation.

Setting `enabled: false` or `mode_override: auto` forces autonomous execution regardless of capability, eliminating probe and teacher costs entirely. The `fallback_mode` determines behavior when probing fails or times out.

The probe uses a fast, cheap model since it runs before every task. Probe latency adds directly to task startup time, so fast models (Grok Fast, Gemini Flash) minimize this overhead.

## Storage Configuration

Storage persists execution traces, rewards, and learning data to PostgreSQL. Without storage, learning cannot function since playbook entries require database persistence.

```yaml
storage:
  database_url: postgresql://user:pass@host:port/dbname
  min_connections: 1
  max_connections: 5
  statement_timeout_seconds: 30
```

Connection pooling parameters control concurrency. Single-agent workflows work with min=1, max=5. Multi-agent or high-throughput deployments benefit from higher limits. The `statement_timeout_seconds` prevents hung queries from blocking the pool.

## Configuration Strategies

Budget optimization uses Gemini Flash across all components except the agent. This reduces RIM and learning costs while maintaining agent quality. Disabling adaptive teaching eliminates probe overhead. Increasing RIM variance threshold reduces large model escalations.

```yaml
agent:
  llm:
    model: gpt-4.1-mini

rim:
  small_model:
    model: gemini/gemini-2.5-flash
  large_model:
    model: gemini/gemini-2.5-flash
  variance_threshold: 0.25

learning:
  llm:
    model: gemini/gemini-2.5-flash
  gates:
    max_text_length: 500

adaptive_teaching:
  enabled: false
```

Quality optimization uses stronger models and tighter thresholds. Lower variance threshold increases RIM consistency through more frequent large model arbitration. Higher auto threshold ensures teacher validation on borderline tasks. Increased learning text length allows more detailed guidance.

```yaml
agent:
  llm:
    model: o1-preview

teacher:
  llm:
    model: gpt-5

rim:
  small_model:
    model: gemini/gemini-2.5-pro
  large_model:
    model: gemini/gemini-2.5-pro
  variance_threshold: 0.10

learning:
  llm:
    model: gemini/gemini-2.5-pro
  gates:
    max_text_length: 800

adaptive_teaching:
  probe:
    thresholds:
      auto: 0.90
```

Latency optimization uses fast models and reduces intervention frequency. Disabling probes eliminates pre-task overhead. Reducing learning history limit triggers synthesis less often. Lowering teacher thresholds increases auto mode usage.

The dominant cost factors are agent calls (1-5 per task), RIM calls (2-4 per task), and teacher calls (0-3 per task depending on mode). Learning synthesis runs every N tasks where N equals `history_limit`, so its per-task cost amortizes across the window. Probe cost is negligible since fast models handle capability assessment.

## Performance Metrics

Track configuration impact through session metrics. Cost per task reflects LLM calls across agent, teacher, and RIM. Latency per task includes model inference plus orchestration overhead. Token usage determines billing under per-token pricing.

```sql
SELECT
  AVG(total_cost) as avg_cost,
  AVG(duration_seconds) as avg_duration,
  AVG(total_tokens) as avg_tokens
FROM sessions
WHERE created_at > NOW() - INTERVAL '7 days';
```

Learning adoption rate measures how often playbook entries inject into prompts. Low adoption suggests cues are too specific or entries fail to match actual tasks. High adoption with poor performance improvement indicates low-quality entries, suggesting tighter gate rules or adjusted rubric weights.

```sql
SELECT
  COUNT(*) FILTER (WHERE metadata->>'applied_student_learning' IS NOT NULL) * 100.0 / COUNT(*) as adoption_pct
FROM sessions
WHERE created_at > NOW() - INTERVAL '7 days';
```

Execution mode distribution shows how often the system operates autonomously versus requiring teacher intervention. High auto mode percentage indicates strong agent capability or loose thresholds. High coach mode percentage suggests aggressive thresholds or weak agent performance.

```sql
SELECT
  metadata->>'execution_mode' as mode,
  COUNT(*) as count
FROM sessions
WHERE created_at > NOW() - INTERVAL '7 days'
GROUP BY mode;
```
