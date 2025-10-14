# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapted from NeMo Agent Toolkit data_models.config."""

from __future__ import annotations

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

class OpenAIAdapterConfig(AdapterConfig):
    """Adapter that proxies OpenAI compatible chat completions."""

    type: Literal[AdapterType.OPENAI] = AdapterType.OPENAI
    llm: LLMParameters
    response_format: Dict[str, Any] | None = None

    @field_validator("llm")
    @classmethod
    def ensure_openai_provider(cls, value: LLMParameters):
        if value.provider not in {LLMProvider.OPENAI, LLMProvider.AZURE_OPENAI}:
            raise ValueError("openai adapter requires an OpenAI compatible provider")
        return value

AdapterUnion = HTTPAdapterConfig | PythonAdapterConfig | OpenAIAdapterConfig

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

AdaptiveMode = Literal["auto", "paired", "coach", "escalate"]


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

    llm: "LLMParameters | None" = None
    thresholds: AdaptiveProbeThresholds = Field(default_factory=AdaptiveProbeThresholds)
    fallback_mode: Literal["paired", "coach", "escalate"] = "paired"
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

class StorageConfig(BaseModel):
    """PostgreSQL connection settings."""

    model_config = ConfigDict(extra="forbid")

    database_url: str
    min_connections: int = Field(default=1, ge=1)
    max_connections: int = Field(default=5, ge=1)
    statement_timeout_seconds: float = Field(default=30.0, ge=0.0)

class AtlasConfig(BaseModel):
    """Root configuration consumed by the Atlas SDK."""

    model_config = ConfigDict(extra="forbid")

    agent: AdapterUnion = Field(discriminator="type")
    student: StudentConfig = Field(default_factory=StudentConfig)
    teacher: TeacherConfig
    orchestration: OrchestrationConfig = Field(default_factory=OrchestrationConfig)
    rim: RIMConfig
    storage: StorageConfig | None = None
    prompt_rewrite: PromptRewriteConfig | None = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    adaptive_teaching: AdaptiveTeachingConfig = Field(default_factory=AdaptiveTeachingConfig)
