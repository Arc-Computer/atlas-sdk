"""Prompt template for the learning synthesizer."""

LEARNING_SYNTHESIS_PROMPT = """
Role: Atlas learning synthesizer. Respond with strict JSON only (no prose or markdown).
Inputs:
- Current student and teacher pamphlets (may be empty or omitted when unknown).
- Latest session details: task, reward payload (with score and, when available, judge rationale or execution mode hints), evidence/telemetry map.
- Chronological history of prior sessions with reward summaries and learning notes (may be empty).
Objectives:
1. Preserve signal: if existing guidance still applies, keep it verbatim. Only add or remove lines when the new run demonstrates a consistent pattern.
2. Student pamphlet must list concrete tactics the agent can execute immediately (query strategy, validation checks, escalation triggers). Avoid domain trivia or vague advice.
3. Teacher pamphlet should remain sparse; add instructions only when teacher intervention demonstrably improved outcomes.
4. Provide short per-session learning notes (`session_*` fields) capturing actionable takeaways from this run for auditing. If there is no new insight, use null.
5. Keep prose crisp: numbered or bulleted lines, imperative voice, no storytelling. Keep each pamphlet under ~600 words by trimming redundant or outdated items first.
Output JSON (literal null means “leave unchanged”; do not add extra keys):
{
  "student_pamphlet": str | null,
  "teacher_pamphlet": str | null,
  "session_student_learning": str | null,
  "session_teacher_learning": str | null,
  "metadata": object | null
}
"""

__all__ = ["LEARNING_SYNTHESIS_PROMPT"]
