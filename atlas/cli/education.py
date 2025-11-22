"""Just-in-time mental model education for atlas env init.

Provides concise explanations of Atlas concepts for first-time users.
"""

import os
from pathlib import Path
from typing import Optional


def is_first_run(config_dir: Optional[str] = None) -> bool:
    """Check if this is the user's first run of atlas env init.

    Args:
        config_dir: Optional config directory path (defaults to .atlas)

    Returns:
        True if first run, False otherwise
    """
    if config_dir is None:
        config_dir = ".atlas"

    marker_file = Path(config_dir) / ".initialized"
    return not marker_file.exists()


def mark_initialized(config_dir: Optional[str] = None):
    """Mark that atlas env init has been run.

    Args:
        config_dir: Optional config directory path (defaults to .atlas)
    """
    if config_dir is None:
        config_dir = ".atlas"

    config_path = Path(config_dir)
    config_path.mkdir(parents=True, exist_ok=True)

    marker_file = config_path / ".initialized"
    marker_file.write_text("# Atlas SDK initialized\n")


def print_welcome_message():
    """Print welcome message for first-time users."""
    print("\nWelcome to Atlas SDK!")
    print(
        "Atlas wraps your existing agent with continual learning capabilities.\n"
    )


def print_core_concepts():
    """Print concise explanation of core Atlas concepts."""
    print("Core Concepts:")
    print()
    print("  Student = Your agent (executes tasks)")
    print("    • Learns from experience over time")
    print("    • Receives persistent playbook guidance")
    print("    • Improves with few-shot examples")
    print()
    print("  Teacher = Validation layer (ensures quality)")
    print("    • Reviews outputs for correctness")
    print("    • Provides guidance when needed")
    print("    • Adapts supervision based on confidence")
    print()
    print("  Playbook = Persistent learnings (improves performance)")
    print("    • Extracted from successful sessions")
    print("    • Injected into prompts automatically")
    print("    • Grows over time with experience")
    print()
    print("  Probe = Smart router (selects execution mode)")
    print("    • auto: Student only (high confidence)")
    print("    • paired: Student + Teacher validation (medium)")
    print("    • coach: Full review + guidance (low confidence)")
    print()


def print_execution_modes():
    """Print explanation of execution modes."""
    print("\nExecution Modes:")
    print()
    print("  auto (fastest)")
    print("    ├─ Student executes task independently")
    print("    └─ No teacher validation overhead")
    print()
    print("  paired (balanced)")
    print("    ├─ Student executes task")
    print("    ├─ Teacher validates final output")
    print("    └─ Retry if validation fails")
    print()
    print("  coach (thorough)")
    print("    ├─ Teacher reviews plan first")
    print("    ├─ Student executes with guidance")
    print("    ├─ Teacher validates each step")
    print("    └─ Multiple retry opportunities")
    print()


def print_learning_features():
    """Print explanation of learning features."""
    print("\nLearning Features (enabled by default):")
    print()
    print("  Playbook Injection")
    print("    • Persistent learnings from past sessions")
    print("    • Automatically injected into prompts")
    print("    • Helps student avoid past mistakes")
    print()
    print("  Few-Shot Prompting")
    print("    • Real examples of successful behaviors")
    print("    • Extracted from session history")
    print("    • Shows concrete cue → action patterns")
    print()
    print("  Adaptive Teaching")
    print("    • Probe selects mode based on task confidence")
    print("    • Reduces supervision overhead over time")
    print("    • Balances speed and quality")
    print()


def print_data_flow():
    """Print explanation of data flow and storage."""
    print("\nData Flow:")
    print()
    print("  Session → Storage → Learning → Playbook → Student")
    print()
    print("  1. Session: Task execution with student/teacher")
    print("  2. Storage: PostgreSQL records all interactions")
    print("  3. Learning: LLM synthesizes persistent playbooks")
    print("  4. Playbook: Guidance injected into future prompts")
    print("  5. Student: Improved performance on similar tasks")
    print()


def print_next_steps_education():
    """Print guidance on next steps after initialization."""
    print("\nWhat happens when you run 'atlas run'?")
    print()
    print("  1. Probe analyzes task and selects execution mode")
    print("  2. Student (your agent) executes task with playbook")
    print("  3. Teacher validates output (if mode requires)")
    print("  4. Reward evaluator computes session quality")
    print("  5. Learning synthesizer generates new playbook entries")
    print("  6. Session stored in PostgreSQL for review")
    print()
    print("Over time:")
    print("  • Playbook grows with domain-specific guidance")
    print("  • Student performance improves on similar tasks")
    print("  • Probe learns when supervision is needed")
    print("  • Reward scores increase as agent gets better")
    print()


def print_architecture_link():
    """Print link to architecture documentation."""
    print("\nLearn More:")
    print("  Architecture: https://docs.arc-computer.com/atlas-sdk")
    print("  Examples: ./examples/")
    print("  CLAUDE.md: Complete architecture reference")
    print()


def show_first_time_education():
    """Show complete first-time user education."""
    print_welcome_message()
    print_core_concepts()
    print_learning_features()
    print()
    print("─" * 60)
    print()


def show_quick_reference():
    """Show quick reference for returning users."""
    print("\nQuick Reference:")
    print("  Student = your agent")
    print("  Teacher = validation layer")
    print("  Playbook = persistent learnings")
    print("  Probe = mode selector (auto/paired/coach)")
    print()


def should_show_education(config_dir: Optional[str] = None) -> bool:
    """Determine if educational content should be shown.

    Args:
        config_dir: Optional config directory path

    Returns:
        True if should show education, False otherwise
    """
    # Show education on first run
    return is_first_run(config_dir)


def show_education_if_needed(config_dir: Optional[str] = None):
    """Show educational content if this is a first run.

    Args:
        config_dir: Optional config directory path
    """
    if should_show_education(config_dir):
        show_first_time_education()
        mark_initialized(config_dir)
    else:
        # Returning users get a brief reminder
        pass  # Could show quick reference if desired


def print_configuration_explanation(anthropic_only: bool = True):
    """Explain configuration choices.

    Args:
        anthropic_only: Whether using Anthropic-only configuration
    """
    if anthropic_only:
        print("\nConfiguration Choice:")
        print("  Based on runtime evaluation benchmarks:")
        print("    • Claude 4.5 Haiku (Student): Fast, cost-effective")
        print("    • Claude 4.5 Sonnet (Teacher): Powerful, accurate")
        print("    • Performance: 0.989 reward, 20.08s latency")
        print()
        print("  You can customize this later in the generated config file.")
        print()
