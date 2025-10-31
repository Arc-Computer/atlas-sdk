"""Learning Harness for MCP Tool Usage Optimization

This script runs 20+ learning episodes to demonstrate how Atlas SDK helps agents
improve their tool usage efficiency over time. It tracks metrics like:
- Number of tool calls per task
- Task completion rate
- Reward progression
- Tool selection accuracy
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from atlas.core import arun
from atlas.config.models import AtlasConfig


# Progressive task complexity: from simple to complex operations
LEARNING_TASKS = [
    # Phase 1: Basic file operations (tasks 1-5)
    "List all files in the sample_workspace directory",
    "Read the contents of sample_workspace/notes.txt",
    "Create a new file called hello.txt with the content 'Hello World'",
    "List files again and verify hello.txt was created",
    "Read the hello.txt file you just created",

    # Phase 2: Multi-step operations (tasks 6-10)
    "Read notes.txt and create a copy called notes_backup.txt with the same content",
    "Search for the word 'important' in all files in sample_workspace",
    "Create a new file summary.txt that lists all files you found",
    "Read multiple files: notes.txt, hello.txt, and summary.txt",
    "Search for files containing the word 'backup'",

    # Phase 3: Complex workflows (tasks 11-15)
    "List all files, then read each .txt file and create a combined file called combined.txt",
    "Search for the word 'test' in sample_workspace and write results to search_results.txt",
    "Create a file called manifest.txt listing all files with their approximate sizes",
    "Read manifest.txt and create a new file top_files.txt with just the first 3 entries",
    "Search for any files containing numbers and document your findings in numbers_found.txt",

    # Phase 4: Advanced scenarios (tasks 16-20)
    "Create a backup of all .txt files by reading each and writing to a _backup version",
    "Use the list_files tool to find all files, then search each for the word 'backup'",
    "Read all backup files and create a single file called all_backups.txt with their combined content",
    "Create a file called report.txt that summarizes: total files, files with 'backup', files with 'test'",
    "Perform a cleanup: list all files, identify which are backups, and document this in cleanup_plan.txt",

    # Phase 5: Edge cases and error handling (tasks 21-25)
    "Try to read a non-existent file called missing.txt and handle the error gracefully",
    "Search for a complex regex pattern '\\d{3}' in all files",
    "Create a file with a very long name: this_is_a_test_file_with_a_very_long_name_to_test_limits.txt",
    "List files in a non-existent directory and handle the error",
    "Perform a complete workspace audit: list all files, check their contents, and create audit_log.txt",
]


async def run_learning_session():
    """Execute the learning harness with 25 progressive tasks."""

    print("=" * 80)
    print("MCP Tool Learning Harness - Atlas SDK")
    print("=" * 80)
    print(f"\nStarting learning session with {len(LEARNING_TASKS)} tasks")
    print(f"Tasks progress from simple file operations to complex multi-step workflows\n")

    # Load configuration
    config_path = Path(__file__).parent / "config.yaml"

    # Track metrics across runs
    session_results = []
    start_time = datetime.now()

    for idx, task in enumerate(LEARNING_TASKS, 1):
        print(f"\n{'=' * 80}")
        print(f"Task {idx}/{len(LEARNING_TASKS)}: {task}")
        print(f"{'=' * 80}")

        try:
            # Run the task with Atlas SDK (using arun for async context)
            result = await arun(
                task=task,
                config_path=str(config_path),
                stream_progress=True,  # Stream output for visibility
            )

            # Extract metrics from result
            status = getattr(result, 'status', 'unknown')
            session_id = getattr(result, 'session_id', None)

            session_results.append({
                'task_number': idx,
                'task': task,
                'status': status,
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            })

            print(f"\n✓ Task {idx} completed: {status}")

            # Brief pause between tasks to avoid rate limits
            if idx < len(LEARNING_TASKS):
                await asyncio.sleep(1)

        except Exception as e:
            print(f"\n✗ Task {idx} failed: {e}")
            session_results.append({
                'task_number': idx,
                'task': task,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })

    # Print summary
    duration = datetime.now() - start_time
    print("\n" + "=" * 80)
    print("Learning Session Complete")
    print("=" * 80)

    successful = sum(1 for r in session_results if r.get('status') == 'succeeded')
    failed = sum(1 for r in session_results if r.get('status') == 'failed')

    print(f"\nTotal tasks: {len(LEARNING_TASKS)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Duration: {duration}")

    # Save results summary
    results_file = Path(__file__).parent / f"learning_results_{start_time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'session_count': len(LEARNING_TASKS),
            'successful': successful,
            'failed': failed,
            'duration_seconds': duration.total_seconds(),
            'results': session_results
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    # Instructions for viewing learning playbook
    print("\n" + "=" * 80)
    print("Next Steps: Analyzing Learning Progress")
    print("=" * 80)
    print("\n1. View the learning playbook and synthesis:")
    print("   python -m atlas.cli.learning --project mcp-tool-learning")
    print("\n2. Export session traces for further analysis:")
    print("   arc-atlas --database-url postgresql://atlas:atlas@localhost:5433/atlas \\")
    print("             --output mcp_traces.jsonl --limit 25")
    print("\n3. Check the Postgres database to see reward progression:")
    print("   psql postgresql://atlas:atlas@localhost:5433/atlas")
    print("   SELECT session_id, task, reward_score FROM atlas_sessions ORDER BY session_id DESC LIMIT 25;")

    return session_results


if __name__ == "__main__":
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it before running: export OPENAI_API_KEY=sk-...")
        sys.exit(1)

    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set")
        print("Please set it before running: export GEMINI_API_KEY=...")
        sys.exit(1)

    # Run the learning session
    asyncio.run(run_learning_session())
