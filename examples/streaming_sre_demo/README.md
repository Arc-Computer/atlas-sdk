## Streaming SRE Demo (Adaptive Runtime)

This demo shows how to run the continual-learning runtime with the new triage → probe →
adaptive flow for an incident-response workflow.

### 1. Configure the runtime

`configs/examples/http_agent.yaml` now includes a default triage adapter and adaptive teaching settings:

```yaml
adaptive_teaching:
  triage_adapter: atlas.utils.triage:default_build_dossier
  certify_first_run: true
  default_tags:
    - tenant:sre-demo
    - domain:sre
```

Copy the config and adjust:

```bash
cp configs/examples/http_agent.yaml configs/local_sre.yaml
```

Edit `adaptive_teaching.triage_adapter` if you want to plug in a custom dossier builder
(generate one via `atlas triage init --domain sre --output sre_dossier.py`).

### 2. Run the demo task

```bash
python examples/http_example.py \
  --config configs/local_sre.yaml \
  --task "Resolve TLS certificate mismatch on payments-router"
```

Key behaviours to observe:

- First unseen fingerprint → `paired` certification; teacher verdict reused as reward.
- Subsequent runs with history → capability probe selects `coach`/`auto`.
- Persona metadata (`helpful_count`, `last_mode`, tags) updates automatically.

### 3. Inspect telemetry / exports

Console output now includes adaptive summaries:

```
Adaptive: mode=coach confidence=0.58
  probe -> mode=coach confidence=0.55
  probe evidence: persona_helpful_ratio=0.62, risk_high_severity
```

Export the session for dashboards:

```bash
arc-atlas \
  --database-url postgresql://localhost:5432/atlas \
  --output sre-demo.jsonl \
  --session-id <SESSION_ID>
```

The JSONL record will contain `adaptive_summary`, `triage_dossier`, `personas_used`, and
`persona_updates`, matching the contract in `docs/runtime_adaptive_flow.md`.
