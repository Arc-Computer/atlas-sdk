"""Prompt templates used by reward judges."""

PROCESS_PROMPT = (
    "Role: Execution trajectory quality judge. Apply the supplied principles.\n"
    "Task: Evaluate how well the student executed the plan with the teacher's guidance.\n\n"
    "Context:\n"
    "- Step Description: {step_description}\n"
    "- Dependencies: {dependencies}\n"
    "- Teacher Guidance: {guidance}\n"
    "- Execution Trace: {student_trace}\n"
    "- Final Output: {student_output}\n\n"
    "Step 1: Generate 3 or 4 principles tailored to this execution. Each principle must include\n"
    "         a short name, a weight between 0.0 and 1.0 (weights sum to 1.0), and a one-sentence description.\n"
    "         Consider alignment to plan, clarity of reasoning, and correct tool usage.\n"
    "Step 2: Score the execution against every principle.\n"
    "Step 3: Deliver a final score in [0,1] with a rationale referencing each principle.\n"
    "Report an uncertainty value in [0,1]; use >0.3 when unsure.\n\n"
    "Return JSON: {{\"principles\": [{{\"name\": str, \"weight\": float, \"description\": str}}],\n"
    "              \"score\": float, \"rationale\": str, \"uncertainty\": float}}."
)

HELPFULNESS_PROMPT = (
    "Role: Teaching effectiveness judge. Apply the supplied principles.\n"
    "Task: Evaluate whether the teacher's guidance improved the student's outcome.\n\n"
    "Context:\n"
    "- Task: {task}\n"
    "- Teacher Guidance: {teacher_trace}\n"
    "- Student Execution Trace: {student_trace}\n"
    "- Final Output: {student_output}\n"
    "- Prior Step Results: {prior_results}\n\n"
    "Step 1: Generate 3 or 4 principles for assessing helpfulness. Consider specificity, actionability, coverage of\n"
    "         failure modes, and connection to the student's final result. Provide a name, weight (sum to 1.0), and\n"
    "         short description for each principle.\n"
    "Step 2: Score the teaching quality against each principle while reflecting on the student's outcome.\n"
    "Step 3: Provide a final score in [0,1] with a rationale referencing every principle.\n"
    "Include an uncertainty value in [0,1]; use >0.3 when confidence is low.\n\n"
    "Return JSON: {{\"principles\": [{{\"name\": str, \"weight\": float, \"description\": str}}],\n"
    "              \"score\": float, \"rationale\": str, \"uncertainty\": float}}."
)

__all__ = ["PROCESS_PROMPT", "HELPFULNESS_PROMPT"]
