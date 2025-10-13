# Continual Learning Runtime Blueprint

**Last updated:** 2025-10-12  
**Audience:** Runtime / RL / DX engineering teams  
**Goal:** Provide an authoritative description of the adaptive capability controller and reward-driven memory system that powers Atlas SDK’s continual-learning flywheel.

**Product principles (first principles recap)**

1. **Keep the customer’s agent fast and inexpensive when it already knows what to do.**
2. **Capture rich guidance only when the agent genuinely needs help.**
3. **Feed those trajectories back into persona memories / world-model training loops.**

The architecture below operationalizes these principles: the capability probe keeps strong agents in the fast lane, the heavy guidance path only triggers on novel or risky incidents, and the reward-driven memory system turns every intervention into structured data for continual learning and downstream training.

### Quick-start summary (for BYOA teams)

1. **Register your agent** in the config (import path + callable + optional system prompt).
2. **Specify the reward objective** (or accept the default evaluator).
3. **Optional:** add domain tags / tweak adaptive settings.
4. **Run the runtime** – the controller handles intervention automatically (first unseen fingerprint runs a one-time `paired` certification pass unless `adaptive_teaching.certify_first_run` is disabled).
5. **Export data** (CLI) straight into Atlas Core (or your own trainer) when you want to retrain a teacher/world model.

---

## 1. Why We Need This (Memo Alignment)

`memo.md` promises:

- The **system** (not the base model) is the product.
- Customers get a **continual-learning harness** that compounds into a proprietary world model.
- We deliver a **J-curve** on performance: early guidance is costly, but cost drops as the agent learns.
- The **reward system** is more than a scalar—it steers learning.

To honor that vision, our runtime has to:

1. **Let capable agents run fast** (low latency, low cost, no handholding).
2. **Capture rich guidance only when needed** and ensure it updates the agent’s memory.
3. **Make every run contribute** to the customer’s world model (structured, retrievable knowledge).

Past demos (e.g., streaming SRE) exposed gaps:

- Teacher always rewrote the plan and expected tool artefacts.
- BYOA agents returning structured JSON failed validation → zero reward.
- Persona promotion never triggered; telemetry gave no insight.

The adaptive controller + reward-driven memory system fixes this.

---

## 2. High-Level Architecture

```
Triaged Dossier  ─┐
                  ├─► Capability Probe ─► Adaptive Mode (auto / paired / coach / escalate)
Persona Stats    ─┘

Adaptive Mode ─► Plan + Validation Behaviour (student executes accordingly)
                 
Execution Transcript ─► Reward Model ─► Memory Updates (per persona / fingerprint)
                                │
                                ├─► Telemetry & Exports (mode, confidence, used memories)
                                └─► Persona Promotion weighting
```

Key principles:

- **Diagnose before acting.** Determine how much help the student needs.
- **Adapt mode per run.** Fast lanes (auto, paired, coach) vs heavy lane (escalate).
- **Reward controls learning.** Rich transcript feeds the reward system, which updates persistent memory.
- **Structured memory.** Guidance is stored as addressable entries, not a single document.

---

## 3. Adaptive Modes (Execution Tiers)

| Mode | When selected | Plan behaviour | Validation | Cost profile |
|------|----------------|----------------|------------|--------------|
| `auto` | Known fingerprint, high reward history | Keep student plan; force `single_shot` | Skip intermediate validation | ≈ single model call |
| `paired` | Medium confidence | Keep plan; teacher can append brief note | Validate final answer only | Slight overhead |
| `coach` | Uncertain but recoverable | Allow short plan (≤2 steps) | Teacher sends short coaching reflection; optional follow-up | Moderate |
| `escalate` | Novel + low reward / high-risk | Full plan rewrite, retries, persona generation | Teacher validates each step, issues guidance | Highest |

**The system evolves to spend more time in auto/paired** as memory improves.

---

## 4. Core Runtime Changes

### 4.1 Capability Probe & Triage

- **Triage dossier** (structured context) becomes first-class (e.g., incident summary + signals).
- `Teacher.adaptive_probe()` consumes dossier + recent metrics + persona stats. For fingerprints with **no history**, the controller bypasses the probe and forces an initial certification pass (see below).
- Probe returns JSON: `{ "mode": "auto", "confidence": 0.82, "evidence": [...], "recommended_personas": [...] }` for fingerprints with existing history.
- Stored in `ExecutionContext.metadata["adaptive_probe"]`.
- **Model/runtime behaviour:**
  - Default probe model: GPT‑4o‑mini / Claude Haiku (configurable per deployment).
  - Probe timeout or very low confidence → default to safe `coach` mode.
  - Student can raise a `help_needed` signal mid-run; the controller escalates to `coach`/`escalate` if confidence drops.
- **Starter adapters:** We ship reference triage builders for incident response, customer support, and code workflows. A CLI scaffold (`atlas triage init`) helps new domains generate a skeleton adapter.

### 4.2 Adaptive Controller

- `atlas/runtime/orchestration/orchestrator.py` reads probe output.
- If the fingerprint has **no reward history**, the controller forces `paired` mode (one-time certification pass). The teacher’s verification is reused as the initial reward signal. This behaviour is configurable via `adaptive_teaching.certify_first_run`.
- For fingerprints with history, forces `single_shot` for auto/paired/coach; only escalate uses multi-step.
- Retry policy & telemetry respect the mode.

### 4.3 Teacher Behaviour

- `Teacher.areview_plan` & `Teacher.avalidate_step` branch on the mode.
- Auto: keep plan, skip step validation.
- Paired: final-answer validation only.
- Coach: short reflection; no tool output required.
- Escalate: current behaviour (multi-step, guidance, retries).
- Certification runs (first-time `paired`) reuse the teacher’s verdict as the primary reward and skip redundant evaluator calls.

### 4.4 Execution Context & Telemetry

- `ExecutionContext.metadata` now preserves:
  - `triage.dossier` – the exact dossier emitted by the adapter.
  - `adaptive` – probe payload, mode history, certification flag.
  - `adaptive_summary` – flattened view consumed by console/exports.
  - `applied_persona_memories` – memory IDs used during the run.
- Persona metadata (helpful/harmful counts, last mode/reward, system tags) is updated in place so promotion and telemetry have a consistent schema.
- `ConsoleTelemetryStreamer` surfaces adaptive mode, probe confidence/evidence, and recent decisions at the end of each run.
- `arc-atlas` JSONL exports now include:
  - `adaptive_summary`
  - `triage_dossier`
  - `personas_used`
  - `persona_updates`
  - `teacher_notes`
  - `reward_summary`

See `docs/adaptive_runtime_guide.md` for the full CLI flow and payload structure.

---

## 5. Reward & Memory System

We ingest the full transcript (student attempts + teacher hints + outcome).

- Reward model assesses success/failure and provides reasons.
- **Structured memory entries** per persona/fingerprint (IDs, tags, success rate) are updated:
  - Helpful entries get reinforced.
  - Harmful entries flagged and possibly pruned.
  - New entries added for novel guidance.
- Promotion weighting depends on mode: `auto` success ≈ strong evidence; `escalate` success ≈ evidence but flagged as teacher-assisted.

### 5.1 Memory entry schema

Each persona/fingerprint stores a list of entries with consistent metadata so the reward system can mark them helpful/harmful and downstream tooling can audit changes.

```json
{
  "fingerprint": "sre/mtls-cert-chain",
  "entry_id": "ca32b5f4-...",
  "source_session": 231,
  "created_at": "2025-10-12T14:05:21Z",
  "status": "active | candidate | retired",
  "text": "Validate SPIFFE IDs and reload payment-router bundle before retrying mTLS handshake.",
  "tags": ["teacher_guidance", "tls"],
  "helpful_count": 4,
  "harmful_count": 0,
  "last_mode_used": "paired",
  "last_reward": 0.94
}
```

- **Tags:**
  - The runtime automatically assigns system tags such as `teacher_guidance`, `student_self_note`, `auto_mode_success`, etc., based on how the entry was produced.
  - Users may optionally supply custom tags (e.g. `tenant:sre-demo`, `component:tls`) via config or when writing BYOA integrations. These get merged with system tags.
  - The reward system updates `helpful_count` / `harmful_count` and may attach derived tags (`reward:low`, `reward:high`).
- **Status transitions:** Controlled by the runtime (promotion/demotion); users can force-retire entries via API if needed.

This mirrors ReasoningBank/ACE:
- Acute logging of which entries were used.
- Reflection tags for each entry (helpful/harmful/neutral).
- Incremental updates avoid prompt collapse.

---

## 6. Continual Learning Loop

1. **Auto lane success** → reward logs a clean victory; persona usage stats roll up to high confidence.
2. **Escalate lane** → reward writes new hints; next time, the probe should flip to coach/paired.
3. **Telemetry** shows how many tasks stay in `auto` vs `escalate` (health metric).
4. **DX gets a simple story**: “if your agent is strong, it stays in auto and runs cheap; when it learns, it graduates to auto.”

This is the J-curve: initial escalations are expensive, but days later most traffic drops into auto/paired.

---

## 7. Developer Experience & Customer Value

- **BYOA simplicity:** Provide the triage dossier + agent output; the controller handles mode. No need to fabricate tool outputs.
- **Cost visibility:** Telemetry shows how much time you spend in each mode.
- **World model accumulation:** Customers can export their structured memory (the persona entries) as part of their “world model asset.”
- **Safe defaults:** Config defaults to `adaptive_teaching.enabled = true`. Users can override, but they get the best behaviour out-of-the-box.

### 7.1 User configuration surface

| Area | What the user sets | Default behaviour |
|------|--------------------|-------------------|
| **Student agent** | Import path, callable function, adapter settings (API keys, working dir, etc.). | None – user must supply their agent. |
| **Reward objective** | Plug in reward function or adjust evaluator weights/thresholds. | Built-in evaluator (pass/fail scoring) if not overridden. |
| **Reasoning level / intervention** | Optional override (`adaptive_teaching.mode_override = auto|coach|escalate`) or tweak probe thresholds. | Runtime runs a one-time `paired` certification pass, then runs capability probe and selects mode automatically. |
| **Memory curation** | Optionally seed, retire, or tag entries (e.g., via reporting helper or direct DB edit). | Runtime promotes/demotes based on reward + thresholds; auto-tags entries. |
| **Telemetry/Exports** | Choose export cadence, configure CLI output path, forward to downstream trainer. | JSONL export command ships with sane defaults; telemetry stream always on. |
| **Custom tags** | Provide domain/tenant tags to attach to new entries. | System tags (`teacher_guidance`, `student_self_note`, etc.) always applied. |

**Reasoning level note:** Users do **not** need to micromanage “low/medium/high reasoning.” The capability probe handles tier selection. Overriding is available for regulated workflows (e.g., force `escalate` while certifying a new agent) but not required for day-to-day continual learning.

---

## 8. File/Module Impact (Implementation Guide)

| Component | Key files | Notes |
|-----------|-----------|-------|
| Triage & capability probe | `atlas/personas/teacher.py`, new utils in `atlas/utils/triage.py` | Add prompt + helper for dossiers |
| Controller & execution | `atlas/runtime/orchestration/orchestrator.py`, `.../execution_context.py` | Manage mode flow |
| Teacher prompts | `atlas/prompts/*` | New prompts referencing modes |
| Reward updates | `atlas/evaluation/evaluator.py`, `atlas/runtime/persona_memory/*` | Tag entries, update metadata |
| Telemetry/export | `atlas/runtime/telemetry/*`, `atlas/cli/jsonl_writer.py` | Additional fields |
| Config/docs | `configs/examples/*`, `examples/streaming_sre_demo/README.md`, `docs/` | Document usage |

---

## 9. Performance & Cost Considerations

- **Latency:** Capability probe ≈ O(1). In auto lanes, runtime ≈ single model call. This is a huge win over always running multi-step flows.
- **Token usage:** Escalate still costs, but becomes rare as memory improves.
- **Scaling:** Reward-driven updates keep memory entries targeted, so we avoid prompt bloat and “context collapse.”
- **J-curve:** Telemetry should show increasing fraction of runs in auto/paired; this is the KPI for continual-learning maturity. Certification runs (first encounter) are expected to incur one extra teacher verification before subsequent auto/paired runs drop latency dramatically.
- **Dashboards:** Provide a default dashboard (or CLI summary) showing adaptive_mode distribution, latency tokens per run, and persona promotions so customers can see efficiency improvements immediately.

---

## 10. Related Work Anchors

- **ReasoningBank:** structured memory, use failures, incremental updates.
- **ACE:** generator/reflector/curator pipeline → we adopt those roles across student/teacher/reward.
- **Hierarchical Memories:** anchor model + retrieval bank → student is anchor, personas/hints are memories.

We implement their principles with Atlas tooling: adaptive controller + reward-managed personas.

---

## 11. Rollout Phases

1. **Phase 0:** Implement capability probe + adaptive controller. Nudge streaming SRE demo to use it.
2. **Phase 1:** Wire reward model to update structured memory entries (helpful/harmful tags) and adjust promotion weighting.
3. **Phase 2:** Replace task-tag fingerprinting with triage-derived embeddings.
4. **Phase 3:** Build dashboards/KPIs for mode distribution and persona promotions.

---

## 12. Final Notes

- The continual-learning harness hinges on three pillars: **triage → adaptive execution → reward-driven memory.**
- Each run either confirms existing knowledge or enriches it with targeted guidance.
- This is the route to the memo’s world-model vision: a system that compounds a customer’s proprietary operational intelligence—cheap when the agent is strong, precise when it needs to learn.
