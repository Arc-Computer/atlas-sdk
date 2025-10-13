# Adaptive Controller End-to-End Flow

```mermaid
flowchart LR
    subgraph Input
        A[Raw user request / task]
        B[Optional telemetry (metrics, diffs, customer profile)]
    end

    A --> C[Triage Adapter]
    B --> C
    C --> D[Triage Dossier (structured JSON)]

    subgraph Mode Selection
        D --> E{History exists?}
        E -->|no| F1[Force paired (certification)]
        E -->|yes| F2[Capability Probe (LLM or rule-based)]
        F2 -->|mode, confidence, evidence| F{Adaptive Mode}
        F1 --> G
        F -->|auto| G1
        F -->|paired| G2
        F -->|coach| G3
        F -->|escalate| G4
    end

    subgraph Execution
        G[F1 certification run] --> H
        G1[Single-shot execution] --> H
        G2[Single-shot + final review] --> H
        G3[Single-shot + coaching reflection] --> H
        G4[Multi-step plan + retries] --> H
        H[Execution transcript (student + teacher)] --> I[Reward Model]
    end

    I -->|judgement, annotations| J[Memory Update (persona entries)]
    J --> K[Telemetry & JSONL Export]
    K --> L[Atlas Core / custom trainer]
    J --> M[Runtime persona cache]

    M --> E
    note right of F1: Teacher verdict reused as initial reward (configurable via certify_first_run)
    note right of D: Stored in ExecutionContext
```

## Runtime Components

- **Triage Adapter** – Implemented via `atlas.utils.triage:default_build_dossier` or a
  custom adapter generated with `atlas triage init`. Outputs are persisted in
  `ExecutionContext.metadata["triage"]["dossier"]`.
- **Capability Probe** – `Teacher.adaptive_probe` consumes the dossier and persona stats,
  emitting mode, confidence, evidence, and recommended personas. Probe payloads are
  cached under `ExecutionContext.metadata["adaptive"]["probe"]`.
- **Adaptive Controller** – The orchestrator records mode history, enforces
  certify-first-run (paired certification), and routes execution:
  - `auto` → single shot, validation skipped.
  - `paired` → single shot, certification verdict reused as reward override.
  - `coach` → stepwise with short guidance and reduced retries.
  - `escalate` → full stepwise supervision and retries.
- **Reward & Memory** – Certification verdicts bypass the evaluator, persona metadata
  is updated with helpful/harmful counts, last mode, and tags, and usage records capture
  the adaptive outcome.
- **Telemetry & Export** – Console output and JSONL export now include
  `adaptive_summary`, probe evidence, persona usage, and teacher notes. See
  `docs/adaptive_runtime_guide.md` for CLI examples.
