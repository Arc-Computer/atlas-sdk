"""
Atlas SDK Python adapter for SecRL (ExCyTIn-Bench).

Simple bridge between Atlas and SecRL - writes metadata to ExecutionContext.
"""
import json
import os
import sys
from typing import Any, Dict, Sequence

# Add SecRL to path
SECRL_ROOT = os.path.dirname(os.path.abspath(__file__))
if SECRL_ROOT not in sys.path:
    sys.path.insert(0, SECRL_ROOT)

from secgym.excytin_env import ExcytinEnv
from secgym.evaluator import LLMEvaluator
from secgym.agents import BaselineAgent
from atlas.runtime.orchestration.execution_context import ExecutionContext, ExecutionContextState


def _build_plan_payload(task: str | None, config: Dict[str, Any]) -> str:
    """Produce a minimal single-step plan in JSON."""
    description = "Interrogate the SecRL environment to answer the security incident question, then synthesize findings."
    if task:
        description = f"Address the task: {task.strip()}. {description}"
    plan = {
        "steps": [
            {
                "id": 1,
                "description": description,
                "tool": None,
                "tool_params": {
                    "attack": config["attack"],
                    "question_index": config["question_index"],
                    "max_steps": config["max_steps"],
                    "layer": config["layer"],
                },
                "depends_on": [],
            }
        ]
    }
    return json.dumps(plan)


def _resolve_config(metadata: Dict[str, Any], session_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Combine session metadata and call metadata for SecRL configuration."""
    def _lookup(key: str, default: Any) -> Any:
        if isinstance(metadata, dict) and key in metadata:
            return metadata[key]
        if key in session_meta:
            return session_meta[key]
        return default

    attack = _lookup("attack", "incident_55")
    question_index = int(_lookup("question_index", 0) or 0)
    max_steps = int(_lookup("max_steps", 15) or 15)
    layer = _lookup("layer", "alert")
    return {
        "attack": attack,
        "question_index": question_index,
        "max_steps": max_steps,
        "layer": layer,
    }


def _summarise_artifacts(session_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Provide non-spoiler telemetry for the teacher while keeping answers hidden."""
    artifacts: Dict[str, Any] = {}
    for key in ("secrl_steps_used", "secrl_attack"):
        if key in session_meta:
            artifacts[key] = session_meta[key]
    if "secrl_error" in session_meta:
        artifacts["secrl_error"] = session_meta["secrl_error"]
    if "secrl_timeout" in session_meta:
        artifacts["secrl_timeout"] = session_meta["secrl_timeout"]
    return artifacts


def _run_secrl(
    config: Dict[str, Any],
    llm_config: Dict[str, Any],
    session_meta: Dict[str, Any],
    guidance: Sequence[str] | None,
    guidance_digest: str,
    learning_pamphlet: str | None = None,
) -> str:
    """Execute the SecRL baseline agent and populate session metadata."""
    model = llm_config.get("model", "gpt-4.1-nano")
    api_key_env = llm_config.get("api_key_env", "OPENAI_API_KEY")
    api_key = os.getenv(api_key_env)

    if model.startswith("gemini/"):
        clean_model = model.replace("gemini/", "")
        agent_config = [{
            "model": clean_model,
            "api_key": api_key,
            "temperature": llm_config.get("temperature", 0.0),
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        }]
        eval_config = [{
            "model": clean_model,
            "api_key": api_key,
            "temperature": 0.0,
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        }]
    else:
        agent_config = [{
            "model": model,
            "api_key": api_key,
        }]
        eval_config = [{
            "model": model,
            "api_key": api_key,
        }]

    if "timeout_seconds" in llm_config:
        agent_config[0]["timeout"] = llm_config["timeout_seconds"]

    if "max_output_tokens" in llm_config:
        if model.startswith("gemini/"):
            agent_config[0]["max_tokens"] = llm_config["max_output_tokens"]
        else:
            agent_config[0]["max_completion_tokens"] = llm_config["max_output_tokens"]

    agent_temperature = 0.0 if model.startswith("gemini/") else 1

    agent = BaselineAgent(
        config_list=agent_config,
        cache_seed=41,
        max_steps=config["max_steps"],
        temperature=agent_temperature,
    )

    evaluator = LLMEvaluator(
        config_list=eval_config,
        cache_seed=42,
        ans_check_reflection=True,
        sol_check_reflection=True,
        step_checking=False,
        strict_check=False,
    )

    env = ExcytinEnv(
        attack=config["attack"],
        evaluator=evaluator,
        save_file=True,
        max_steps=config["max_steps"],
        split="test",
        use_full_db=False,
        layer=config["layer"],
    )

    observation, info = env.reset(config["question_index"])
    if learning_pamphlet:
        session_meta["secrl_learning_pamphlet"] = learning_pamphlet
        pamphlet_block = "\nLearning Guidance:\n" + learning_pamphlet.strip()
        observation = f"{observation}\n\n{pamphlet_block}"
    if guidance:
        guidance_block = "\nTeacher Guidance:\n" + "\n".join(f"- {g}" for g in guidance)
        observation = f"{observation}\n\n{guidance_block}"
    agent.reset()

    final_answer = None
    trajectory = []

    try:
        for step_num in range(config["max_steps"]):
            try:
                action, submit = agent.act(observation)

                trajectory.append({
                    "step": step_num + 1,
                    "observation": observation,
                    "action": action,
                    "submit": submit,
                })

                observation, reward, done, info = env.step(action=action, submit=submit)

                if submit:
                    final_answer = info.get("answer", action)
                    session_meta["secrl_reward"] = reward
                    session_meta["secrl_correct"] = (reward == 1.0)
                    session_meta["secrl_evaluation"] = info
                    break

                if done:
                    break

            except Exception as exc:  # pragma: no cover - defensive guard
                error_msg = f"Error at step {step_num + 1}: {exc}"
                trajectory.append({"error": error_msg})
                final_answer = f"Error during execution: {exc}"
                session_meta["secrl_error"] = error_msg
                break

        if final_answer is None:
            final_answer = "Agent did not submit an answer within the step limit."
            session_meta["secrl_timeout"] = True

        session_meta["secrl_trajectory"] = trajectory
        session_meta["secrl_steps_used"] = len(trajectory)
        session_meta["secrl_question"] = env.curr_question
        session_meta["secrl_attack"] = config["attack"]
        if isinstance(env.curr_question, dict) and "solution" in env.curr_question:
            session_meta["secrl_solution"] = env.curr_question["solution"]
        if guidance:
            session_meta["secrl_applied_guidance"] = list(guidance)

        agent_logs = agent.get_logging()
        session_meta["secrl_usage"] = agent_logs.get("usage_summary", {})

        session_meta["secrl_final_answer"] = final_answer
        session_meta["secrl_guidance_digest"] = guidance_digest
        return final_answer
    finally:
        try:
            env.close()
        except Exception:  # pragma: no cover - best effort cleanup
            pass


async def main(prompt: str, metadata: dict) -> str:
    """
    Atlas adapter entry point - simple bridge to SecRL.

    Args:
        prompt: Security question to answer
        metadata: Config from Atlas (llm_config provided by SDK)

    Returns:
        Agent's answer to the security question or structured JSON during planning/execution.
    """
    context = ExecutionContext(ExecutionContextState.get())
    session_meta = context.metadata.setdefault("session_metadata", {})
    session_meta.setdefault("secrl_task_prompts", []).append(prompt)

    call_mode = metadata.get("mode") if isinstance(metadata, dict) else None
    config = _resolve_config(metadata, session_meta)

    steps_meta = context.metadata.get("steps", {})
    step_id = 1
    if isinstance(metadata, dict):
        candidate_step = metadata.get("step_id")
        if isinstance(candidate_step, int):
            step_id = candidate_step
    guidance_entries: list[str] = []
    last_attempt_valid: bool | None = None
    if isinstance(steps_meta, dict):
        step_record = steps_meta.get(step_id)
        if isinstance(step_record, dict):
            raw_guidance = step_record.get("guidance")
            if isinstance(raw_guidance, list):
                for item in raw_guidance:
                    if isinstance(item, str):
                        cleaned = item.strip()
                        if cleaned:
                            guidance_entries.append(cleaned)
            attempts = step_record.get("attempts")
            if isinstance(attempts, list) and attempts:
                last_attempt = attempts[-1]
                evaluation = last_attempt.get("evaluation")
                if isinstance(evaluation, dict):
                    validation = evaluation.get("validation")
                    if isinstance(validation, dict):
                        flag = validation.get("valid")
                        if isinstance(flag, bool):
                            last_attempt_valid = flag
    guidance_digest = "|".join(guidance_entries)
    last_guidance_digest = session_meta.get("secrl_guidance_digest", "")
    cached_answer = session_meta.get("secrl_final_answer")
    has_cached_answer = isinstance(cached_answer, str) and cached_answer.strip()

    learning_state = context.metadata.get("learning_state")
    learning_pamphlet: str | None = None
    if isinstance(learning_state, dict):
        session_meta["learning_state"] = learning_state
        student_learning = learning_state.get("student_learning")
        if isinstance(student_learning, str) and student_learning.strip():
            learning_pamphlet = student_learning.strip()

    if call_mode == "planning":
        task = context.metadata.get("task") if isinstance(context.metadata.get("task"), str) else prompt
        return _build_plan_payload(task, config)

    llm_config = metadata.get("llm_config", {})

    if call_mode == "synthesis":
        if has_cached_answer and guidance_digest == last_guidance_digest and (last_attempt_valid is True or last_attempt_valid is None):
            return cached_answer
        final_answer = _run_secrl(
            config,
            llm_config,
            session_meta,
            guidance_entries,
            guidance_digest,
            learning_pamphlet,
        )
        return final_answer or "No answer produced by SecRL baseline agent."

    # Execution mode (default): always re-run SecRL so teacher guidance can affect retries.
    if has_cached_answer and guidance_digest == last_guidance_digest and (last_attempt_valid is True or last_attempt_valid is None):
        artifacts = _summarise_artifacts(session_meta)
        payload = {
            "status": "completed",
            "result": {
                "deliverable": cached_answer,
                "artifacts": artifacts,
            },
        }
        return json.dumps(payload)

    final_answer = _run_secrl(
        config,
        llm_config,
        session_meta,
        guidance_entries,
        guidance_digest,
        learning_pamphlet,
    )

    artifacts = _summarise_artifacts(session_meta)
    payload = {
        "status": "completed",
        "result": {
            "deliverable": final_answer,
            "artifacts": artifacts,
        },
    }
    return json.dumps(payload)
