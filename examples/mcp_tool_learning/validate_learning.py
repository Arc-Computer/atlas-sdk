"""Quick validation run for MCP tool learning (5 tasks)."""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from atlas.core import arun

# Run 5 tasks for validation
TASKS = [
    "List all files in the sample_workspace directory",
    "Read the contents of sample_workspace/notes.txt",
    "Create a new file called hello.txt with the content 'Hello World'",
    "Read the hello.txt file you just created",
    "Search for the word 'important' in all files in sample_workspace",
]


async def main():
    config_path = Path(__file__).parent / "config.yaml"
    
    print("=" * 80)
    print("MCP Tool Learning Validation (5 tasks)")
    print("=" * 80)
    
    # Change to the mcp_tool_learning directory for proper imports
    original_cwd = Path.cwd()
    mcp_dir = Path(__file__).parent
    
    try:
        os.chdir(mcp_dir)
        
        for idx, task in enumerate(TASKS, 1):
            print(f"\nTask {idx}/5: {task}")
            try:
                result = await arun(
                    task=task,
                    config_path=str(config_path),
                    stream_progress=True,
                    session_metadata={
                        "learning_key_override": "mcp-tool-learning-validation",
                        "incident_id": f"mcp-task-{idx}",
                    },
                )
                print(f"✓ Task {idx} completed")
            except Exception as e:
                print(f"✗ Task {idx} failed: {e}")
                return 1
    finally:
        os.chdir(original_cwd)
    
    print("\n✅ Validation complete!")
    print("   Learning key: mcp-tool-learning-validation")
    print("   Run validation script: python scripts/validate_empirical_learning.py mcp-tool-learning-validation")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

