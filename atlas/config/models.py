# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapted from NeMo Agent Toolkit data_models.config."""

from __future__ import annotations

import warnings
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Sequence

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

warnings.filterwarnings(
    "ignore",
    message='Field name "schema" in "LearningConfig" shadows an attribute in parent "BaseModel"',
    category=UserWarning,
)

class RetryPolicy(BaseModel):
    """Retry behavior for adapter calls."""

    model_config = ConfigDict(extra="forbid")

    attempts: int = Field(default=1, ge=1, le=5)
    backoff_seconds: float = Field(default=1.0, ge=0.0)

class ToolParameterSchema(BaseModel):
    """JSON schema describing tool parameters."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["object"] = Field(default="object")
    properties: Dict[str, Any] = Field(default_factory=dict)
    required: List[str] = Field(default_factory=list)
    additionalProperties: bool = Field(default=False, alias="additionalProperties")

    @field_validator("required")
    @classmethod
    def ensure_required_keys_exist(cls, value: List[str], info):
        if not value:
            return value
        missing = [key for key in value if key not in info.data.get("properties", {})]
        if missing:
            joined = ", ".join(sorted(missing))
            raise ValueError(f"required fields missing from properties: {joined}")
        return value

class ToolDefinition(BaseModel):
    """Defines a callable tool exposed to the Student."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: ToolParameterSchema = Field(default_factory=ToolParameterSchema)
    output_schema: Dict[str, Any] | None = Field(default=None, alias="outputSchema")

class AdapterType(str, Enum):
    """Adapter implementations supported by Atlas."""

    HTTP = "http_api"
    PYTHON = "python"
    OPENAI = "openai"
    LITELLM = "litellm"

class AdapterConfig(BaseModel):
    """Base configuration shared by BYOA adapters."""

    model_config = ConfigDict(extra="forbid")

    type: AdapterType
    name: str
    system_prompt: str
    tools: List[ToolDefinition] = Field(default_factory=list)

class HTTPAdapterTransport(BaseModel):
    """Connection parameters for HTTP adapters."""

    model_config = ConfigDict(extra="forbid")

    base_url: str
    headers: Dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=60.0, ge=0.0)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)

class HTTPAdapterConfig(AdapterConfig):
    """Adapter using an HTTP endpoint."""

    type: Literal[AdapterType.HTTP] = AdapterType.HTTP
    transport: HTTPAdapterTransport
    payload_template: Dict[str, Any] = Field(default_factory=dict)
    result_path: Sequence[str] | None = None

class PythonAdapterConfig(AdapterConfig):
    """Adapter wrapping a Python callable."""

    type: Literal[AdapterType.PYTHON] = AdapterType.PYTHON
    import_path: str
    attribute: str | None = None
    working_directory: str | None = None
    allow_generator: bool = False
    llm: "LLMParameters | None" = None

class LLMProvider(str, Enum):
    """LLM providers supported by Atlas."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure-openai"
    BEDROCK = "bedrock"
    GOOGLE = "google"
    GEMINI = "gemini"
    XAI = "xai"

class MetadataDigestConfig(BaseModel):
    """Controls how execution metadata is projected into LLM-facing prompts.

    Defaults reserve roughly 10%% of the provider's published context window, assuming ~4 characters per token.
    Override ``char_budget`` or ``provider_char_budgets`` if you need a different policy.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    char_budget: int | None = Field(default=None, ge=1024)
    provider_char_budgets: Dict[LLMProvider, int] = Field(default_factory=dict)
    include_session_keys: List[str] = Field(
        default_factory=lambda: [
            "source",
            "execution_mode",
            "adaptive_summary",
            "token_usage",
            "reward_summary",
            "reward_stats",
            "reward_audit_summary",
            "triage_dossier",
            "drift_alert",
            "learning_usage",
            "student_learning",
            "teacher_learning",
            "session_learning_note",
            "notes",
        ]
    )
    max_plan_steps: int = Field(default=5, ge=0, le=20)
    max_step_summaries: int = Field(default=5, ge=0, le=20)
    max_learning_history_entries: int = Field(default=3, ge=0, le=10)
    max_reward_audit_entries: int = Field(default=3, ge=0, le=10)
    max_prompt_rewrite_chars: int = Field(default=2000, ge=256, le=20000)
    max_section_chars: int = Field(default=4000, ge=512, le=20000)
    max_string_chars: int = Field(default=1000, ge=128, le=4000)


class LLMParameters(BaseModel):
    """Configuration for an LLM request path."""

    model_config = ConfigDict(extra="forbid")

    provider: LLMProvider = LLMProvider.OPENAI
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    api_base: str | None = None
    organization: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_output_tokens: int | None = Field(default=None, ge=1)
    timeout_seconds: float = Field(default=60.0, ge=0.0)
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    additional_headers: Dict[str, str] = Field(default_factory=dict)
    reasoning_effort: Literal["low", "medium", "high"] | None = None

class LitellmAdapterConfig(AdapterConfig):
    """Multi-provider adapter proxying chat completions via litellm."""

    type: Literal[AdapterType.LITELLM] = AdapterType.LITELLM
    llm: LLMParameters
    response_format: Dict[str, Any] | None = None
    metadata_digest: MetadataDigestConfig = Field(default_factory=MetadataDigestConfig)

    @field_validator("llm")
    @classmethod
    def ensure_litellm_provider(cls, value: LLMParameters):
        allowed = {
            LLMProvider.OPENAI,
            LLMProvider.AZURE_OPENAI,
            LLMProvider.ANTHROPIC,
            LLMProvider.GEMINI,
            LLMProvider.BEDROCK,
            LLMProvider.XAI,
        }
        if value.provider not in allowed:
            raise ValueError(
                "litellm adapter requires an OpenAI-compatible provider (OpenAI/Azure) or LiteLLM-supported provider "
                f"({', '.join(provider.value for provider in allowed if provider not in {LLMProvider.OPENAI, LLMProvider.AZURE_OPENAI})})"
            )
        return value


class OpenAIAdapterConfig(AdapterConfig):
    """Deprecated: Use LitellmAdapterConfig instead.
    
    This class is maintained for backward compatibility. New code should use
    LitellmAdapterConfig with type: litellm.
    """

    type: Literal[AdapterType.OPENAI] = AdapterType.OPENAI
    llm: LLMParameters
    response_format: Dict[str, Any] | None = None
    metadata_digest: MetadataDigestConfig = Field(default_factory=MetadataDigestConfig)

    @field_validator("llm")
    @classmethod
    def ensure_openai_provider(cls, value: LLMParameters):
        warnings.warn(
            "OpenAIAdapterConfig is deprecated. Use LitellmAdapterConfig with type: litellm instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # Duplicate validation logic (cannot call static method from another class)
        allowed = {
            LLMProvider.OPENAI,
            LLMProvider.AZURE_OPENAI,
            LLMProvider.ANTHROPIC,
            LLMProvider.GEMINI,
            LLMProvider.BEDROCK,
            LLMProvider.XAI,
        }
        if value.provider not in allowed:
            raise ValueError(
                "litellm adapter requires an OpenAI-compatible provider (OpenAI/Azure) or LiteLLM-supported provider "
                f"({', '.join(provider.value for provider in allowed if provider not in {LLMProvider.OPENAI, LLMProvider.AZURE_OPENAI})})"
            )
        return value

AdapterUnion = HTTPAdapterConfig | PythonAdapterConfig | LitellmAdapterConfig | OpenAIAdapterConfig

class StudentPrompts(BaseModel):
    """Prompt templates used when delegating to the Student."""

    model_config = ConfigDict(extra="forbid")

    planner: str
    executor: str
    synthesizer: str

class TeacherPrompts(BaseModel):
    """Prompt templates used to derive teacher personas."""

    model_config = ConfigDict(extra="forbid")

    plan_review: str
    validation: str
    guidance: str

AdaptiveMode = Literal["auto", "paired", "coach"]


class AdaptiveProbeThresholds(BaseModel):
    """Confidence thresholds that map capability probe scores to execution modes."""

    model_config = ConfigDict(extra="forbid")

    auto: float = Field(default=0.85, ge=0.0, le=1.0)
    paired: float = Field(default=0.65, ge=0.0, le=1.0)
    coach: float = Field(default=0.35, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_order(self) -> "AdaptiveProbeThresholds":
        if not (self.auto >= self.paired >= self.coach):
            raise ValueError("confidence thresholds must satisfy auto ≥ paired ≥ coach")
        return self


class AdaptiveProbeConfig(BaseModel):
    """Settings that control capability probe behaviour."""

    model_config = ConfigDict(extra="forbid")

    llm: "LLMParameters | None" = Field(
        default_factory=lambda: LLMParameters(
            provider=LLMProvider.XAI,
            model="xai/grok-4-fast",
            api_key_env="XAI_API_KEY",
            temperature=0.2,
            timeout_seconds=20.0,
        )
    )
    thresholds: AdaptiveProbeThresholds = Field(default_factory=AdaptiveProbeThresholds)
    fallback_mode: Literal["paired", "coach"] = "paired"
    evidence_limit: int = Field(default=6, ge=1, le=32)
    timeout_seconds: float = Field(default=15.0, ge=1.0)


class RewardObjectiveConfig(BaseModel):
    """Allows BYOA deployments to override the default reward objective."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["rim", "python"] = "rim"
    import_path: Optional[str] = None
    attribute: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: float | None = Field(default=None, ge=0.0)
    focus_prompt: str | None = None

    @model_validator(mode="after")
    def _validate_python_target(self) -> "RewardObjectiveConfig":
        if self.type == "python" and not self.import_path:
            raise ValueError("reward.import_path is required when type='python'")
        return self


class AdaptiveTeachingConfig(BaseModel):
    """Global adaptive-teaching controls for the runtime."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    certify_first_run: bool = True
    mode_override: AdaptiveMode | None = None
    triage_adapter: str | None = None
    default_tags: List[str] = Field(default_factory=list)
    probe: AdaptiveProbeConfig = Field(default_factory=AdaptiveProbeConfig)
    reward: RewardObjectiveConfig = Field(default_factory=RewardObjectiveConfig)
    learning_history_limit: int = Field(default=10, ge=1, le=200)

    @field_validator("default_tags", mode="before")
    @classmethod
    def _coerce_tags(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            tags = [str(item).strip() for item in value if str(item).strip()]
            return tags
        return [str(value).strip()] if str(value).strip() else []


class PromptRewriteConfig(BaseModel):
    """Controls how persona prompts are derived via LLM."""

    model_config = ConfigDict(extra="forbid")

    llm: LLMParameters | None = None
    max_tokens: int = Field(default=1024, ge=64)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)

class StudentConfig(BaseModel):
    """Configuration for the Student wrapper."""

    model_config = ConfigDict(extra="forbid")

    prompts: StudentPrompts | None = None
    prompt_guidance: Dict[str, str] = Field(default_factory=dict)
    max_plan_tokens: int = Field(default=2048, ge=1)
    max_step_tokens: int = Field(default=2048, ge=1)
    max_synthesis_tokens: int = Field(default=2048, ge=1)
    tool_choice: Literal["auto", "required"] = "auto"

class TeacherConfig(BaseModel):
    """Configuration for plan review and guidance."""

    model_config = ConfigDict(extra="forbid")

    llm: LLMParameters
    max_review_tokens: int | None = Field(default=None, ge=1)
    plan_cache_seconds: int = Field(default=300, ge=0)
    guidance_max_tokens: int | None = Field(default=None, ge=1)
    validation_max_tokens: int | None = Field(default=None, ge=1)
    prompts: TeacherPrompts | None = None
    prompt_guidance: Dict[str, str] = Field(default_factory=dict)

class LearningPrompts(BaseModel):
    """Prompt templates for the learning synthesizer."""

    model_config = ConfigDict(extra="forbid")

    synthesizer: str | None = None


class PlaybookEntryGateRules(BaseModel):
    """Gate configuration enforced on generated learning playbook entries."""

    model_config = ConfigDict(extra="forbid")

    enforce_actionability: bool = True
    enforce_cue: bool = True
    enforce_generality: bool = False  # Default False - use empirical validation instead
    max_text_length: int = Field(default=420, ge=100, le=2000)
    allowed_proper_nouns: List[str] = Field(
        default_factory=lambda: ["SQL", "HTTP", "JSON", "Atlas", "API"]
    )
    banned_incident_tokens: List[str] = Field(
        default_factory=lambda: ["incident", "ticket", "case", "postmortem"]
    )
    allow_length_overflow_margin: int = Field(default=20, ge=0, le=500)


class PlaybookEntryRubricWeights(BaseModel):
    """Weighted scoring rubric applied to learning playbook entries."""

    model_config = ConfigDict(extra="forbid")

    actionability: float = Field(default=0.4, ge=0.0, le=1.0)
    generality: float = Field(default=0.3, ge=0.0, le=1.0)
    hookability: float = Field(default=0.2, ge=0.0, le=1.0)
    concision: float = Field(default=0.1, ge=0.0, le=1.0)


class PlaybookPruningConfig(BaseModel):
    """Configuration for empirical pruning of playbook entries."""

    model_config = ConfigDict(extra="forbid")

    min_sessions: int = Field(default=10, ge=1, le=100)
    min_cue_hit_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    min_reward_delta: float = Field(default=0.01, ge=-1.0, le=1.0)
    min_transfer_sessions: int = Field(default=20, ge=1, le=100)


class PlaybookEntrySchemaConfig(BaseModel):
    """Schema metadata for playbook entry synthesis."""

    model_config = ConfigDict(extra="forbid")

    version: str = "playbook_entry.v1"
    allowed_runtime_handles: List[str] = Field(default_factory=list)
    runtime_handle_prefixes: List[str] = Field(default_factory=list)
    cue_types: List[str] = Field(default_factory=lambda: ["regex", "keyword", "predicate"])
    default_scope_category: str = "differentiation"
    allow_missing_tool_mapping: bool = False


class LearningUsageConfig(BaseModel):
    """Runtime usage instrumentation toggles."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    capture_examples: bool = False
    max_examples_per_entry: int = Field(default=2, ge=0, le=20)


class LearningConfig(BaseModel):
    """Controls the learning pamphlet synthesizer."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    update_enabled: bool = True
    provisional_acceptance: bool = Field(default=True, description="Accept entries provisionally even if they fail generality gate")
    llm: LLMParameters | None = Field(
        default_factory=lambda: LLMParameters(
            provider=LLMProvider.GEMINI,
            model="gemini/gemini-2.5-flash",
            api_key_env="GEMINI_API_KEY",
            temperature=0.1,
            max_output_tokens=8192,
            timeout_seconds=120.0,
        )
    )
    prompts: LearningPrompts | None = None
    history_limit: int = Field(default=10, ge=1, le=200)
    session_note_enabled: bool = True
    apply_to_prompts: bool = True
    schema: PlaybookEntrySchemaConfig = Field(default_factory=PlaybookEntrySchemaConfig)
    rubric_weights: PlaybookEntryRubricWeights = Field(default_factory=PlaybookEntryRubricWeights)
    gates: PlaybookEntryGateRules = Field(default_factory=PlaybookEntryGateRules)
    pruning_config: PlaybookPruningConfig = Field(default_factory=PlaybookPruningConfig)
    usage_tracking: LearningUsageConfig = Field(default_factory=LearningUsageConfig)

class RIMConfig(BaseModel):
    """Aggregate reward model configuration."""

    model_config = ConfigDict(extra="forbid")

    small_model: LLMParameters
    large_model: LLMParameters
    active_judges: Dict[str, bool] = Field(
        default_factory=lambda: {"process": True, "helpfulness": True}
    )
    variance_threshold: float = Field(default=0.15, ge=0.0)
    uncertainty_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    parallel_workers: int = Field(default=4, ge=1, le=32)
    judge_prompt: str | None = None

class OrchestrationConfig(BaseModel):
    """Controls sequential execution semantics."""

    model_config = ConfigDict(extra="forbid")

    max_retries: int = Field(default=1, ge=0, le=1)
    step_timeout_seconds: float = Field(default=900.0, ge=0.0)
    rim_guidance_tag: str = "rim_feedback"
    emit_intermediate_steps: bool = True
    forced_mode: AdaptiveMode | None = None

class StorageConfig(BaseModel):
    """PostgreSQL connection settings."""

    model_config = ConfigDict(extra="forbid")

    database_url: str
    min_connections: int = Field(default=1, ge=1)
    max_connections: int = Field(default=5, ge=1)
    statement_timeout_seconds: float = Field(default=30.0, ge=0.0)
    apply_schema_on_connect: bool = True


class DriftDetectionConfig(BaseModel):
    """Rolling statistics used to detect reward drift."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    window: int = Field(default=50, ge=1)
    z_threshold: float = Field(default=3.0, ge=0.0)
    min_baseline: int = Field(default=5, ge=0)


class ReviewWorkflowConfig(BaseModel):
    """Review gating applied before exporting traces."""

    model_config = ConfigDict(extra="forbid")

    require_approval: bool = True
    default_export_statuses: List[str] = Field(default_factory=lambda: ["approved"])


class RuntimeSafetyConfig(BaseModel):
    """Top-level guardrail configuration."""

    model_config = ConfigDict(extra="forbid")

    drift: DriftDetectionConfig = Field(default_factory=DriftDetectionConfig)
    review: ReviewWorkflowConfig = Field(default_factory=ReviewWorkflowConfig)

class AtlasConfig(BaseModel):
    """Root configuration consumed by the Atlas SDK."""

    model_config = ConfigDict(extra="forbid")

    agent: AdapterUnion = Field(discriminator="type")
    student: StudentConfig = Field(default_factory=StudentConfig)
    teacher: TeacherConfig
    orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)
    rim: RIMConfig
    learning: LearningConfig = Field(default_factory=LearningConfig)
    storage: StorageConfig | None = None
    prompt_rewrite: PromptRewriteConfig | None = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    adaptive_teaching: AdaptiveTeachingConfig = Field(default_factory=AdaptiveTeachingConfig)
    runtime_safety: RuntimeSafetyConfig = Field(default_factory=RuntimeSafetyConfig)
