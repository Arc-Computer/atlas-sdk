"""Prompt templates for Atlas learning synthesis."""

LEARNING_SYNTHESIS_PROMPT = """
Role: Atlas Learning Synthesizer. Maintain cross-session teaching pamphlets that improve student behaviour and teacher coaching.

Review the provided information and update the persistent pamphlets.

Inputs:
- ``task``: The request the student worked on.
- ``reward``: Summary of the latest reward judgement.
- ``trajectory``: Key steps, guidance, and outputs from the session.
- ``current_pamphlet``: Student/teacher learning guidance currently stored in the registry.
- ``history``: Recent session summaries with rewards and learning notes.

Goals:
1. Distil the core behavioural lesson the student should learn from THIS session. Avoid task-specific facts.
2. Distil any coaching insight for the teacher (null if no intervention or no insight).
3. Update the persistent pamphlet(s) so future sessions start with the best guidance. Preserve useful prior advice while revising or pruning obsolete items.
4. Optionally emit a ``session_note`` when there is a concise observation worth attaching to this single session record.

Output JSON only, exactly with these keys:
{
  "student_learning": str,
  "teacher_learning": str | null,
  "updated_student_pamphlet": str,
  "updated_teacher_pamphlet": str | null,
  "session_note": str | null,
  "metadata": object | null
}

Formatting rules:
- Keep guidance concise, action-oriented, and domain agnostic.
- When returning null, use the literal JSON null value.
- Avoid markdown lists unless already present in the pamphlet.
- Ensure pamphlet fields stay under ~800 words each by consolidating or pruning as needed.
"""

__all__ = ["LEARNING_SYNTHESIS_PROMPT"]
