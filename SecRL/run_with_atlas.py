"""
Example script to run SecRL agent with Atlas SDK.

This demonstrates how Atlas wraps the SecRL baseline agent to provide:
- Adaptive teaching (Teacher validation and guidance)
- Reward signals (RIM judges)
- Learning history (persistent storage)
- Telemetry (structured logging)
"""

import sys
import os

# Add parent directory to path to import atlas
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas import core


def run_single_question(
    attack: str = "incident_5",
    question_index: int = 0,
    max_steps: int = 15,
    stream_progress: bool = True,
):
    """
    Run a single SecRL question through Atlas.

    Args:
        attack: Which incident database to use (incident_5, incident_38, etc.)
        question_index: Which question from the test set to run (0-based)
        max_steps: Maximum agent steps allowed
        stream_progress: Whether to show progress in console
    """
    # Load the SecRL question to use as the task
    # In a real scenario, you might load this from the question file
    task = f"Answer security question {question_index} from {attack}"

    # Run through Atlas
    result = core.run(
        task=task,
        config_path="SecRL/atlas_config.yaml",
        session_metadata={
            "attack": attack,
            "question_index": question_index,
            "max_steps": max_steps,
            "layer": "alert",
        },
        stream_progress=stream_progress,
    )

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Final Answer: {result.final_answer}")
    print(f"Steps Executed: {len(result.step_results)}")

    # Print reward information if available
    for step_result in result.step_results:
        reward = step_result.evaluation.reward
        print(f"\nStep {step_result.step_id} Reward: {reward.score if reward else 'N/A'}")

    return result


def run_multiple_questions(
    attack: str = "incident_5",
    num_questions: int = 5,
    max_steps: int = 15,
):
    """
    Run multiple questions and collect results.

    Args:
        attack: Which incident database to use
        num_questions: How many questions to run
        max_steps: Maximum agent steps per question
    """
    results = []

    for i in range(num_questions):
        print(f"\n{'='*60}")
        print(f"Running Question {i+1}/{num_questions}")
        print(f"{'='*60}\n")

        result = run_single_question(
            attack=attack,
            question_index=i,
            max_steps=max_steps,
            stream_progress=True,
        )

        results.append({
            "question_index": i,
            "final_answer": result.final_answer,
            "num_steps": len(result.step_results),
        })

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for idx, res in enumerate(results, 1):
        print(f"Question {idx}: {res['num_steps']} steps")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SecRL with Atlas SDK")
    parser.add_argument(
        "--attack",
        type=str,
        default="incident_5",
        choices=list(["incident_5", "incident_38", "incident_34", "incident_39",
                      "incident_55", "incident_134", "incident_166", "incident_322"]),
        help="Which incident database to use"
    )
    parser.add_argument(
        "--question",
        type=int,
        default=0,
        help="Question index to run (0-based)"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=1,
        help="Number of questions to run (if >1, runs multiple)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=15,
        help="Maximum agent steps"
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable console streaming"
    )

    args = parser.parse_args()

    # Check if MySQL containers are running
    print("Checking MySQL containers...")
    try:
        import docker
        client = docker.from_env()
        container_name = {
            "incident_5": "incident_5",
            "incident_38": "incident_38",
            "incident_34": "incident_34",
            "incident_39": "incident_39",
            "incident_55": "incident_55",
            "incident_134": "incident_134",
            "incident_166": "incident_166",
            "incident_322": "incident_322",
        }[args.attack]

        container = client.containers.get(container_name)
        if container.status != "running":
            print(f"Warning: Container {container_name} is not running. Starting it...")
            container.start()
            import time
            time.sleep(3)
        print(f"✓ Container {container_name} is running")
    except Exception as e:
        print(f"Warning: Could not verify container status: {e}")
        print("Ensure Docker is running and containers are set up.")

    # Run
    if args.num_questions > 1:
        run_multiple_questions(
            attack=args.attack,
            num_questions=args.num_questions,
            max_steps=args.max_steps,
        )
    else:
        run_single_question(
            attack=args.attack,
            question_index=args.question,
            max_steps=args.max_steps,
            stream_progress=not args.no_stream,
        )
