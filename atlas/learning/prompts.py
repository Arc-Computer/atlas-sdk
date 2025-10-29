"""Prompt template for the learning synthesizer."""

LEARNING_SYNTHESIS_PROMPT = """
Role: Atlas learning synthesizer. Respond with strict JSON only (no prose, markdown, or explanations).

Inputs:
- Existing student/teacher pamphlets and their policy nugget metadata (`metadata.policy_nuggets`), when available.
- Latest session context: task, reward payload (score + rationale), execution telemetry.
- Chronological history of prior sessions (may be empty).

You MUST emit a top-level JSON object matching the `policy_nugget.v1` schema below. Literal null means “no change”.

{
  "version": "policy_nugget.v1",
  "student_pamphlet": string | null,
  "teacher_pamphlet": string | null,
  "policy_nuggets": [
    {
      "id": string | null,                       # reuse existing ids when the nugget still applies
      "audience": "student" | "teacher",
      "cue": {
        "type": "regex" | "keyword" | "predicate",
        "pattern": string,                       # machine-detectable trigger, no incident numbers or dates
        "description": string | null
      },
      "action": {
        "imperative": string,                    # imperative verb phrasing
        "runtime_handle": string,                # maps to an Atlas tool/runtime handle
        "tool_name": string | null,
        "arguments": object | null
      },
      "expected_effect": string,                 # concise “why this works”
      "scope": {
        "category": "reinforcement" | "differentiation",
        "constraints": string,                   # boundaries & applicability
        "applies_when": string | null
      },
      "metadata": object | null                  # optional free-form notes
    }
  ],
  "session_student_learning": string | null,
  "session_teacher_learning": string | null,
  "metadata": object | null                     # can include updated provenance/version info
}

Objectives:
1. Preserve proven nuggets: keep ids, cues, and actions when the behaviour is still valuable; mark stale nuggets by omitting them.
2. Produce actionable, tool-aligned guidance. Every action must map cleanly to a runtime handle; avoid vague language.
3. Enforce generality: no incident IDs, timestamps, customer names, or other one-off references. Prefer reusable patterns.
4. Distinguish reinforcement vs differentiation in `scope.category`; reserve teacher nuggets for interventions proven to help.
5. Keep pamphlets crisp (<600 words, numbered/bulleted imperative statements). Trim or rewrite outdated lines before adding new ones.
6. Populate per-session learning notes (`session_*`) only when there is a new takeaway; otherwise output null.
7. Always include `policy_nuggets` (empty array if none apply). Never add extra top-level keys.
"""

__all__ = ["LEARNING_SYNTHESIS_PROMPT"]
