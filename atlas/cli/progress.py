"""Clean, confident output formatting for atlas env init.

Provides progressive disclosure: clean success output by default,
verbose details available with --verbose flag.
"""

from typing import Dict, List, Optional


class OutputFormatter:
    """Formats output for atlas env init command."""

    def __init__(self, verbose: bool = False):
        """Initialize formatter.

        Args:
            verbose: Whether to show detailed output
        """
        self.verbose = verbose

    def print_header(self, text: str):
        """Print a section header.

        Args:
            text: Header text
        """
        print(f"\n{text}")

    def print_step(self, text: str):
        """Print a progress step.

        Args:
            text: Step description
        """
        print(f"  {text}")

    def print_success(self, label: str, value: str):
        """Print a success item with checkmark.

        Args:
            label: Item label
            value: Item value
        """
        print(f"  ✓ {label}: {value}")

    def print_warning(self, text: str):
        """Print a warning message.

        Args:
            text: Warning text
        """
        print(f"  ⚠ {text}")

    def print_error(self, text: str):
        """Print an error message.

        Args:
            text: Error text
        """
        print(f"  ✗ {text}")

    def print_info(self, label: str, value: str, indent: int = 2):
        """Print an informational item.

        Args:
            label: Item label
            value: Item value
            indent: Number of spaces to indent
        """
        prefix = " " * indent
        print(f"{prefix}• {label}: {value}")

    def print_discovery_phase(self):
        """Print discovery phase header."""
        print("\nScanning repository...")

    def print_discovery_complete(self):
        """Print discovery completion."""
        print("\nDiscovery complete")

    def print_setup_summary(self, summary: Dict[str, str], config_path: str):
        """Print setup summary with key configuration details.

        Args:
            summary: Dictionary of setup items
            config_path: Path to generated config file
        """
        print(f"\n✓ Agent:     {summary.get('Agent', 'unknown')}")
        print(f"✓ Config:    {config_path}")
        print(f"✓ Provider:  {summary.get('Provider', 'unknown')}")

        # Learning details
        learning = summary.get("Learning", "disabled")
        if learning != "disabled":
            print(f"✓ Learning:  {learning}")

        # Storage details
        storage = summary.get("Storage", "not configured")
        if storage != "not configured":
            print(f"✓ Storage:   {storage}")

    def print_mental_model(self):
        """Print concise mental model explanation."""
        print("\nAtlas is configured to:")
        print("  • Wrap your agent as Student (executes tasks)")
        print("  • Add Teacher supervision (validates outputs)")
        print("  • Track learning (improves over time)")
        print("  • Compute rewards (measures quality)")

    def print_validation_results(self, validations: List[Dict[str, bool]]):
        """Print validation results.

        Args:
            validations: List of validation results
        """
        print("\nValidating setup...")
        for validation in validations:
            name = validation["name"]
            passed = validation["passed"]
            if passed:
                print(f"  ✓ {name}")
            else:
                print(f"  ✗ {name}")
                if "error" in validation:
                    print(f"    {validation['error']}")

    def print_confidence(self, confidence: str, issues: Optional[List[str]] = None):
        """Print setup confidence level.

        Args:
            confidence: Confidence level (High/Medium/Low)
            issues: Optional list of issues if confidence is not High
        """
        emoji_map = {
            "High": "✓",
            "Medium": "⚠",
            "Low": "✗",
        }

        emoji = emoji_map.get(confidence, "•")
        print(f"\n{emoji} Setup confidence: {confidence}")

        if issues:
            print("\nIssues:")
            for issue in issues:
                print(f"  • {issue}")

    def print_next_steps(self, config_path: str):
        """Print next steps to run the agent.

        Args:
            config_path: Path to generated config file
        """
        print("\nNext:")
        print(f'  atlas run --config {config_path} --task "Your task here"')

    def print_verbose_details(self, details: Dict[str, any]):
        """Print verbose configuration details (only if verbose=True).

        Args:
            details: Dictionary of detailed information
        """
        if not self.verbose:
            return

        print("\n--- Verbose Configuration Details ---")

        if "candidate_selection" in details:
            sel = details["candidate_selection"]
            print(f"\nCandidate Selection:")
            print(f"  Method: {sel.get('method', 'unknown')}")
            print(f"  Confidence: {sel.get('confidence', 0.0):.2f}")
            print(f"  Reasoning: {sel.get('reasoning', 'N/A')}")

        if "config_decisions" in details:
            print(f"\nConfiguration Decisions:")
            for decision in details["config_decisions"]:
                print(f"  • {decision}")

        if "factory_synthesis" in details:
            print(f"\nFactory Synthesis:")
            print(f"  {details['factory_synthesis']}")

        if "storage_setup" in details:
            print(f"\nStorage Setup:")
            print(f"  {details['storage_setup']}")


def print_init_success(
    candidate_name: str,
    config_path: str,
    summary: Dict[str, str],
    validations: List[Dict[str, bool]],
    confidence: str,
    verbose: bool = False,
    verbose_details: Optional[Dict[str, any]] = None,
):
    """Print complete initialization success output.

    Args:
        candidate_name: Selected agent candidate name
        config_path: Path to generated config file
        summary: Setup summary dictionary
        validations: Validation results
        confidence: Setup confidence level
        verbose: Whether to show verbose output
        verbose_details: Optional verbose details dictionary
    """
    formatter = OutputFormatter(verbose=verbose)

    # Discovery complete
    formatter.print_discovery_complete()

    # Setup summary
    formatter.print_setup_summary(summary, config_path)

    # Mental model (only for first-time users)
    formatter.print_mental_model()

    # Validation results
    formatter.print_validation_results(validations)

    # Confidence
    issues = None
    if confidence != "High":
        issues = [
            v["error"] for v in validations if not v["passed"] and "error" in v
        ]
    formatter.print_confidence(confidence, issues)

    # Next steps
    formatter.print_next_steps(config_path)

    # Verbose details (if requested)
    if verbose and verbose_details:
        formatter.print_verbose_details(verbose_details)


def print_storage_setup_prompt() -> bool:
    """Prompt user to set up persistent storage.

    Returns:
        True if user wants to set up storage, False otherwise
    """
    print("\nPersistent storage not configured.")
    print("Atlas can set up PostgreSQL via Docker for session tracking and learning.")
    response = input("Set up storage now? [Y/n]: ").strip().lower()
    return response in ("", "y", "yes")


def print_storage_setup_progress():
    """Print storage setup progress."""
    print("\nSetting up PostgreSQL storage...")
    print("  • Pulling Docker image")
    print("  • Starting container")
    print("  • Initializing schema")


def print_storage_setup_success(database_url: str):
    """Print storage setup success.

    Args:
        database_url: Database connection URL
    """
    print(f"\n✓ Storage ready: {database_url}")


def print_storage_setup_error(error: str):
    """Print storage setup error.

    Args:
        error: Error message
    """
    print(f"\n✗ Storage setup failed: {error}")
    print("\nYou can set up storage later with:")
    print("  atlas init")
