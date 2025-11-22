"""Default configuration templates for atlas env init.

This module provides opinionated, production-ready defaults based on runtime
evaluation findings. Uses Anthropic-only stack for simplicity.
"""

from typing import Dict, List, Optional

from atlas.config.models import (
    AdaptiveProbeConfig,
    AdaptiveTeachingConfig,
    AtlasConfig,
    LearningConfig,
    LitellmAdapterConfig,
    LLMParameters,
    LLMProvider,
    OrchestrationConfig,
    PythonAdapterConfig,
    RIMConfig,
    RuntimeSafetyConfig,
    StorageConfig,
    StudentConfig,
    TeacherConfig,
    ToolDefinition,
)
from atlas.sdk.discovery import Candidate


def build_anthropic_llm_config(
    model: str,
    temperature: float = 0.2,
    max_output_tokens: int = 4000,
) -> LLMParameters:
    """Build Anthropic LLM configuration.

    Args:
        model: Anthropic model identifier
        temperature: Sampling temperature
        max_output_tokens: Maximum output tokens

    Returns:
        LLMParameters configured for Anthropic
    """
    return LLMParameters(
        provider=LLMProvider.ANTHROPIC,
        model=model,
        api_key_env="ANTHROPIC_API_KEY",
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        timeout_seconds=120.0,
    )


def build_default_config(
    candidate: Candidate,
    tools: Optional[List[ToolDefinition]] = None,
    system_prompt: Optional[str] = None,
    storage_url: Optional[str] = None,
) -> AtlasConfig:
    """Build default Atlas configuration using Anthropic-only stack.

    Based on runtime evaluation findings:
    - Student: Claude 4.5 Haiku (claude-4-5-haiku-20241022)
    - Teacher: Claude 4.5 Sonnet (claude-sonnet-4-5-20250929)
    - Performance: 0.989 reward, 20.08s latency

    Learning features enabled by default:
    - Few-shot prompting (inject_few_shot_examples=True)
    - Playbook injection (playbook_injection_mode="prefix")

    Args:
        candidate: Discovered agent candidate
        tools: Tool definitions for the agent
        system_prompt: Optional custom system prompt
        storage_url: Optional PostgreSQL database URL

    Returns:
        AtlasConfig with Anthropic-only defaults
    """
    # Student LLM: Claude Haiku 4.5 (fast, cost-effective)
    student_llm = build_anthropic_llm_config(
        model="claude-haiku-4-5-20251001",
        temperature=0.2,
        max_output_tokens=4000,
    )

    # Teacher LLM: Claude Sonnet 4.5 (powerful, accurate)
    teacher_llm = build_anthropic_llm_config(
        model="claude-sonnet-4-5-20250929",
        temperature=0.1,
        max_output_tokens=2000,
    )

    # Default system prompt if not provided
    if not system_prompt:
        system_prompt = (
            f"You are an AI assistant wrapped by Atlas SDK for continual learning. "
            f"Your implementation: {candidate.dotted_path()}"
        )

    # Python adapter for BYOA integration
    agent_config = PythonAdapterConfig(
        type="python",
        name=candidate.class_name,
        system_prompt=system_prompt,
        tools=tools or [],
        import_path=candidate.module,
        attribute=candidate.class_name,
    )

    # Student configuration
    student_config = StudentConfig(
        tool_choice="auto",
        max_plan_tokens=2048,
        max_step_tokens=2048,
        max_synthesis_tokens=2048,
    )

    # Teacher configuration
    teacher_config = TeacherConfig(
        llm=teacher_llm,
        plan_cache_seconds=600,
    )

    # RIM configuration (reward evaluation)
    rim_config = RIMConfig(
        small_model=student_llm,
        large_model=teacher_llm,
        variance_threshold=0.15,
        uncertainty_threshold=0.3,
        parallel_workers=4,
    )

    # Learning configuration with few-shot and playbook enabled
    learning_config = LearningConfig(
        enabled=True,
        update_enabled=True,
        llm=student_llm,  # Use Haiku for learning synthesis
        inject_few_shot_examples=True,  # Enable few-shot prompting
        max_few_shot_token_budget=500,
        playbook_injection_mode="prefix",  # Enable playbook injection
        apply_to_prompts=True,
    )

    # Adaptive teaching configuration
    # Use Haiku for probe (fast decision-making)
    probe_llm = build_anthropic_llm_config(
        model="claude-haiku-4-5-20251001",
        temperature=0.2,
        max_output_tokens=1000,
    )

    adaptive_config = AdaptiveTeachingConfig(
        enabled=True,
        probe=AdaptiveProbeConfig(
            llm=probe_llm,
            fallback_mode="paired",
            timeout_seconds=15.0,
        ),
    )

    # Orchestration configuration
    orchestration_config = OrchestrationConfig(
        max_retries=1,
        step_timeout_seconds=900.0,
    )

    # Storage configuration (if URL provided)
    storage_config = None
    if storage_url:
        storage_config = StorageConfig(
            database_url=storage_url,
            min_connections=1,
            max_connections=5,
            statement_timeout_seconds=30.0,
            apply_schema_on_connect=True,
        )

    # Runtime safety configuration
    runtime_safety_config = RuntimeSafetyConfig()

    return AtlasConfig(
        agent=agent_config,
        student=student_config,
        teacher=teacher_config,
        orchestration=orchestration_config,
        rim=rim_config,
        learning=learning_config,
        adaptive_teaching=adaptive_config,
        storage=storage_config,
        runtime_safety=runtime_safety_config,
    )


def get_setup_summary(config: AtlasConfig) -> Dict[str, str]:
    """Generate a human-readable summary of the configuration.

    Args:
        config: Atlas configuration

    Returns:
        Dictionary of summary items for display
    """
    summary = {
        "Agent": config.agent.name,
        "Provider": "Anthropic",
        "Student Model": "Claude Haiku 4.5 (claude-haiku-4-5-20251001)",
        "Teacher Model": "Claude Sonnet 4.5 (claude-sonnet-4-5-20250929)",
    }

    # Learning features
    learning_features = []
    if config.learning.enabled:
        if config.learning.inject_few_shot_examples:
            learning_features.append("few-shot prompting")
        if config.learning.apply_to_prompts:
            learning_features.append(
                f"playbook ({config.learning.playbook_injection_mode})"
            )
        summary["Learning"] = ", ".join(learning_features) if learning_features else "enabled"
    else:
        summary["Learning"] = "disabled"

    # Storage
    if config.storage:
        # Extract host and port from database_url
        url = config.storage.database_url
        if "@" in url and "/" in url:
            host_port = url.split("@")[1].split("/")[0]
            summary["Storage"] = f"PostgreSQL ({host_port})"
        else:
            summary["Storage"] = "PostgreSQL"
    else:
        summary["Storage"] = "not configured"

    # Adaptive teaching
    if config.adaptive_teaching.enabled:
        summary["Adaptive Teaching"] = "enabled"
    else:
        summary["Adaptive Teaching"] = "disabled"

    return summary


def validate_anthropic_key() -> bool:
    """Check if ANTHROPIC_API_KEY is available.

    Returns:
        True if key is present, False otherwise
    """
    import os

    return bool(os.getenv("ANTHROPIC_API_KEY"))
