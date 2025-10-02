"""Prompt templates used by reward judges."""

PROCESS_PROMPT = (
    "You are the Process Judge for the Atlas SDK. Evaluate whether the student's execution trace logically follows"
    " the assigned step. Consider the plan description, the provided trace, and the final output. Respond with a JSON"
    " object containing: 'score' (0 to 1) and 'rationale' (one paragraph)."
)

HELPFULNESS_PROMPT = (
    "You are the Helpfulness Judge for the Atlas SDK. Rate how well the student's step output advances the overall"
    " task. Focus on correctness, completeness, and usefulness. Respond with a JSON object containing: 'score'"
    " (0 to 1) and 'rationale' (one paragraph)."
)

__all__ = ["PROCESS_PROMPT", "HELPFULNESS_PROMPT"]
