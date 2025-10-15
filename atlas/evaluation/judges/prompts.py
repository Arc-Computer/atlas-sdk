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

Step 4: Extract the BEHAVIORAL PATTERN the student should learn (not task-specific content).
Rules for student_learning:
- For FAILURES: Focus on the MISTAKE PATTERN, not the specific task content
- For SUCCESSES: Focus on the EFFICIENT APPROACH that worked, emphasizing how to replicate it faster/cheaper
- Remove all domain-specific terms and make it cross-domain applicable
- Describe the cognitive/procedural pattern, not the content
- Be concrete and actionable for future similar situations

Examples of GOOD student_learning for FAILURES (cross-domain patterns):
✓ "When facing time-sensitive situations with incomplete information, establish a systematic evidence collection process before taking corrective action. Distinguish between symptoms and root causes by mapping dependencies first."
✓ "For tasks involving multiple simultaneous signals or alerts, correlate them by timeline and common attributes before concluding causation. Avoid acting on isolated data points without context."
✓ "When dealing with cascading failures or dependencies, trace impact both upstream and downstream before proposing solutions. Missing indirect effects leads to incomplete remediation."

Examples of GOOD student_learning for SUCCESSES (efficiency patterns):
✓ "For straightforward information synthesis tasks, directly structure the final answer without intermediate reasoning steps. This reduces token usage by 40% while maintaining accuracy."
✓ "When task requirements are explicit and unambiguous, skip exploratory analysis and proceed directly to solution construction. Validate against requirements at the end rather than iteratively."
✓ "For well-defined transformations or formatting tasks, generate the complete output in a single pass rather than building incrementally. This approach is 2-3x faster with equivalent quality."
✓ "When dealing with structured data requests, identify the minimal information set required and query it directly. Avoid over-fetching or exploratory queries that increase latency."

Examples of BAD student_learning:
✗ "Check the firewall logs next time" (domain-specific, not a pattern)
✗ "Answer the question correctly" (too generic)
✗ "Pay more attention to details" (not actionable)
✗ "Good job" (not instructive for future behavior)

Step 5: Extract the TEACHING PATTERN the teacher should learn (not task-specific advice).
Rules for teacher_learning:
- Focus on the PEDAGOGICAL INTERVENTION that worked or failed
- Identify the teaching strategy, not the content taught
- Make it applicable across different domains and situations
- Return null if no teacher intervention occurred

Examples of GOOD teacher_learning (cross-domain strategies):
✓ "When student jumps to conclusions without evidence, ask them to explicitly list evidence supporting AND contradicting their hypothesis. This forces systematic thinking before action."
✓ "If student misses cascading or second-order effects, prompt them to trace dependencies in both directions (what affects this, what does this affect). Don't provide the answer—guide the discovery."
✓ "When student provides overly confident assessments with limited data, ask 'What would change your conclusion?' to help them identify gaps in their reasoning."
✓ "For students who rush to action under time pressure, establish a lightweight decision framework upfront (gather→analyze→decide→act→communicate). Reinforce the framework when they skip steps."

Examples of BAD teacher_learning:
✗ "Tell student to check DNS settings" (domain-specific)
✗ "Guide them better" (too vague)
✗ "Ask more questions" (not specific about what kind)

IMPORTANT: Output JSON only, exactly in this shape and order:
{{"principles": [{{"name": str, "weight": float, "description": str}}],
 "score": float,
 "rationale": str,
 "uncertainty": float,
 "student_learning": str,
 "teacher_learning": str | null}}
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

When synthesizing student_learning and teacher_learning from the tier-1 summaries:
- Extract the BEHAVIORAL PATTERN, not task-specific content
- Remove domain-specific terms to make it cross-domain applicable
- Focus on cognitive/procedural errors for student, pedagogical strategies for teacher
- If tier-1 summaries are task-specific, abstract them to patterns before outputting

Produce the final JSON judgement using the exact schema:
{{"principles": [{{"name": str, "weight": float, "description": str}}],
 "score": float,
 "rationale": str,
 "uncertainty": float,
 "student_learning": str,
 "teacher_learning": str | null}}
"""

__all__ = ["SESSION_REWARD_PROMPT", "SESSION_ARBITER_PROMPT"]
