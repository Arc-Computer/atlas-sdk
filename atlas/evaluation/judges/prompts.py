"""Prompt templates for session-level reward evaluation."""

SESSION_REWARD_PROMPT = """
Role: Expert solution evaluator. Apply the stated instructions precisely.

Task: Evaluate the entire trajectory for the request below.

Context:
- Task / Problem: {task}
- Execution Mode: {execution_mode}
- Teacher Intervened: {teacher_intervened}
- Focus Prompt: {focus_prompt}
- Reviewed Plan: {plan}
- Final Answer: {final_answer}
- Session Metadata: {session_metadata}

For structured outputs (JSON/code), check structural correctness AND semantic accuracy.

Step 1: Derive 2–3 evaluation principles tailored to this trajectory.
Consider (only if supported by evidence):
- Correctness of the final deliverable
- Safety / compliance posture
- Efficiency and retry discipline
- Guidance quality (when teacher intervened)
- Completeness of evidence and tooling use
Give each principle a short name, weight (0.0–1.0, sum to 1.0), and concise description.

Step 2: Evaluate the trajectory against each principle using concrete evidence.

Step 3: Provide the final reward score in [0.0, 1.0] and a rationale explaining the score via those principles.
Report uncertainty in [0.0, 1.0]. Use > 0.3 when evidence is limited or conflicting.

IMPORTANT: Output JSON only, exactly in this shape and order:
{{"principles": [{{"name": str, "weight": float, "description": str}}],
 "score": float,
 "rationale": str,
 "uncertainty": float}}
"""

SESSION_ARBITER_PROMPT = """
Role: Expert session reward arbiter. Resolve disagreements between Tier-1 evaluations.

Context:
- Task / Problem: {task}
- Execution Mode: {execution_mode}
- Teacher Intervened: {teacher_intervened}
- Final Answer: {final_answer}
- Focus Prompt: {focus_prompt}

Tier-1 Summaries:
{tier1_summaries}

Produce the final JSON judgement using the exact schema:
{{"principles": [{{"name": str, "weight": float, "description": str}}],
 "score": float,
 "rationale": str,
 "uncertainty": float}}
"""

__all__ = ["SESSION_REWARD_PROMPT", "SESSION_ARBITER_PROMPT"]
