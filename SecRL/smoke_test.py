"""Smoke test for Atlas + SecRL integration.

Tests incident_5 with first 2 questions.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add parent to path for Atlas import
sys.path.insert(0, str(Path(__file__).parent.parent))

from atlas import core

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


async def run_config(
    config_path: Path,
    learning_key: str,
    label: str,
    start_index: int,
    max_questions: int | None,
) -> list[dict]:
    """Execute the incident_5 set using a specific config and learning key."""

    # Load incident_5 test questions
    questions_file = Path(__file__).parent / "secgym" / "questions" / "test" / "incident_55_qa_incident_o1-ga_c42.json"

    if not questions_file.exists():
        logger.error(f"Questions file not found: {questions_file}")
        logger.info("Available question files:")
        tests_dir = Path(__file__).parent / "secgym" / "questions" / "tests"
        for f in tests_dir.glob("*.json"):
            logger.info(f"  - {f.name}")
        return

    with open(questions_file) as f:
        questions_data = json.load(f)

    # Determine slice of questions to run
    stop_index = len(questions_data) if max_questions is None else min(len(questions_data), start_index + max_questions)
    if start_index >= len(questions_data):
        logger.warning(f"Start index {start_index} exceeds question count ({len(questions_data)}); nothing to do.")
        return []
    questions = questions_data[start_index:stop_index]

    logger.info(f"🔬 Starting smoke test ({label}) for incident_5")
    logger.info(f"📊 Testing {len(questions)} questions (indices {start_index} to {stop_index - 1})")
    logger.info(f"⚙️  Config: {config_path}")

    results = []

    for offset, qa_pair in enumerate(questions):
        idx = start_index + offset
        question = qa_pair.get("question", "")
        expected_answer = qa_pair.get("answer", "")

        logger.info(f"\n{'='*80}")
        logger.info(f"Question {idx + 1}/{len(questions)}")
        logger.info(f"{'='*80}")
        logger.info(f"Q: {question}")
        logger.info(f"Expected: {expected_answer}")
        logger.info(f"{'='*80}\n")

        try:
            # Run through Atlas
            result = await core.arun(
                task=question,
                config_path=str(config_path),
                stream_progress=True,
            session_metadata={
                "attack": "incident_55",
                    "question_index": idx,
                    "max_steps": 15,
                    "layer": "alert",
                    "smoke_test": True,
                    "learning_key_override": learning_key,
                },
            )

            logger.info(f"\n✅ Got result for question {idx + 1}")
            logger.info(f"Answer: {result.final_answer}")

            # Query database for SecRL metadata (it's in sessions.metadata)
            import asyncpg
            conn = await asyncpg.connect("postgresql://atlas:atlas@localhost:5433/atlas")
            try:
                row = await conn.fetchrow(
                    "SELECT metadata FROM sessions ORDER BY created_at DESC LIMIT 1"
                )
                raw_metadata = row["metadata"] if row else None
                if isinstance(raw_metadata, (bytes, bytearray)):
                    raw_metadata = raw_metadata.decode("utf-8")
                if isinstance(raw_metadata, str):
                    try:
                        session_metadata = json.loads(raw_metadata)
                    except json.JSONDecodeError:
                        session_metadata = {}
                elif isinstance(raw_metadata, dict):
                    session_metadata = raw_metadata
                else:
                    session_metadata = {}
                secrl_reward = session_metadata.get('secrl_reward', 'N/A')
                secrl_correct = session_metadata.get('secrl_correct', 'N/A')
            finally:
                await conn.close()

            logger.info(f"SecRL Reward: {secrl_reward}")
            logger.info(f"SecRL Correct: {secrl_correct}")

            results.append({
                "question_index": idx,
                "question": question,
                "expected": expected_answer,
                "actual": result.final_answer,
                "secrl_reward": secrl_reward,
                "secrl_correct": secrl_correct,
                "success": True,
            })

        except Exception as e:
            logger.error(f"\n❌ Error on question {idx + 1}: {e}", exc_info=True)
            results.append({
                "question_index": idx,
                "question": question,
                "expected": expected_answer,
                "error": str(e),
                "success": False,
            })

    return results


async def main(
    config_path: Path,
    learning_key: str,
    label: str,
    start_index: int,
    max_questions: int | None,
) -> bool:
    """Run smoke test for incident_5 using the specified configuration."""

    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return False

    run_results = await run_config(
        config_path,
        learning_key,
        label,
        start_index=start_index,
        max_questions=max_questions,
    )
    successful = sum(1 for r in run_results if r.get("success"))
    total = len(run_results)

    logger.info(f"\n{'='*80}")
    logger.info(f"🎯 SMOKE TEST SUMMARY ({label})")
    logger.info(f"{'='*80}")
    logger.info(f"Total questions: {total}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {total - successful}")

    if successful != total:
        logger.warning(f"\n⚠️  {total - successful} test(s) failed for {label}")

    output_file = Path(__file__).parent / f"smoke_test_results_{label}.json"
    with open(output_file, "w") as f:
        json.dump(run_results, f, indent=2)
    logger.info(f"\n📁 Results ({label}) saved to: {output_file}")

    return successful == total


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run SecRL incident_5 smoke test with a specified config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "atlas_config.yaml",
        help="Path to the atlas configuration file.",
    )
    parser.add_argument(
        "--learning-key",
        type=str,
        default="SecRL_incident_5_shared",
        help="Learning key override to isolate history.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="run",
        help="Label used for logging and output file naming.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="0-based index of the first question to run.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Optional maximum number of questions to run.",
    )
    args = parser.parse_args()

    success = asyncio.run(
        main(
            config_path=args.config,
            learning_key=args.learning_key,
            label=args.label,
            start_index=args.start_index,
            max_questions=args.max_questions,
        )
    )
    sys.exit(0 if success else 1)
