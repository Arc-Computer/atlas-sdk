# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapted from NeMo Agent Toolkit nat.data_models.intermediate_step."""

from __future__ import annotations

import time
import typing
import uuid
from enum import Enum
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

from atlas.runtime.models.invocation_node import InvocationNode


class IntermediateStepCategory(str, Enum):
    LLM = "LLM"
    TOOL = "TOOL"
    WORKFLOW = "WORKFLOW"
    TASK = "TASK"
    FUNCTION = "FUNCTION"
    CUSTOM = "CUSTOM"
    SPAN = "SPAN"


class IntermediateStepType(str, Enum):
    LLM_START = "LLM_START"
    LLM_END = "LLM_END"
    LLM_NEW_TOKEN = "LLM_NEW_TOKEN"
    TOOL_START = "TOOL_START"
    TOOL_END = "TOOL_END"
    WORKFLOW_START = "WORKFLOW_START"
    WORKFLOW_END = "WORKFLOW_END"
    TASK_START = "TASK_START"
    TASK_END = "TASK_END"
    FUNCTION_START = "FUNCTION_START"
    FUNCTION_END = "FUNCTION_END"
    CUSTOM_START = "CUSTOM_START"
    CUSTOM_END = "CUSTOM_END"
    SPAN_START = "SPAN_START"
    SPAN_CHUNK = "SPAN_CHUNK"
    SPAN_END = "SPAN_END"


class IntermediateStepState(str, Enum):
    START = "START"
    CHUNK = "CHUNK"
    END = "END"


class StreamEventData(BaseModel):
    model_config = ConfigDict(extra="allow")
    input: typing.Any | None = None
    output: typing.Any | None = None
    chunk: typing.Any | None = None


class UsageInfo(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    num_llm_calls: int = 0
    seconds_between_calls: float = 0.0


class ToolParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    properties: dict[str, typing.Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)
    type_: Literal["object"] = Field(default="object", alias="type")
    additional_properties: bool = Field(default=False, alias="additionalProperties")


class ToolDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    description: str
    parameters: ToolParameters


class ToolSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["function"]
    function: ToolDetails


class TraceMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")
    chat_responses: typing.Any | None = None
    chat_inputs: typing.Any | None = None
    tool_inputs: typing.Any | None = None
    tool_outputs: typing.Any | None = None
    tool_info: typing.Any | None = None
    span_inputs: typing.Any | None = None
    span_outputs: typing.Any | None = None
    provided_metadata: typing.Any | None = None
    tools_schema: list[ToolSchema] = Field(default_factory=list)


class IntermediateStepPayload(BaseModel):
    model_config = ConfigDict(extra="allow")
    event_type: IntermediateStepType
    event_timestamp: float = Field(default_factory=lambda: time.time())
    span_event_timestamp: float | None = None
    framework: str | None = None
    name: str | None = None
    tags: list[str] | None = None
    metadata: dict[str, typing.Any] | TraceMetadata | None = None
    data: StreamEventData | None = None
    usage_info: UsageInfo | None = None
    UUID: str = Field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def event_category(self) -> IntermediateStepCategory:
        mapping = {
            IntermediateStepType.LLM_START: IntermediateStepCategory.LLM,
            IntermediateStepType.LLM_END: IntermediateStepCategory.LLM,
            IntermediateStepType.LLM_NEW_TOKEN: IntermediateStepCategory.LLM,
            IntermediateStepType.TOOL_START: IntermediateStepCategory.TOOL,
            IntermediateStepType.TOOL_END: IntermediateStepCategory.TOOL,
            IntermediateStepType.WORKFLOW_START: IntermediateStepCategory.WORKFLOW,
            IntermediateStepType.WORKFLOW_END: IntermediateStepCategory.WORKFLOW,
            IntermediateStepType.TASK_START: IntermediateStepCategory.TASK,
            IntermediateStepType.TASK_END: IntermediateStepCategory.TASK,
            IntermediateStepType.FUNCTION_START: IntermediateStepCategory.FUNCTION,
            IntermediateStepType.FUNCTION_END: IntermediateStepCategory.FUNCTION,
            IntermediateStepType.CUSTOM_START: IntermediateStepCategory.CUSTOM,
            IntermediateStepType.CUSTOM_END: IntermediateStepCategory.CUSTOM,
            IntermediateStepType.SPAN_START: IntermediateStepCategory.SPAN,
            IntermediateStepType.SPAN_CHUNK: IntermediateStepCategory.SPAN,
            IntermediateStepType.SPAN_END: IntermediateStepCategory.SPAN,
        }
        return mapping[self.event_type]

    @property
    def event_state(self) -> IntermediateStepState:
        mapping = {
            IntermediateStepType.LLM_START: IntermediateStepState.START,
            IntermediateStepType.LLM_END: IntermediateStepState.END,
            IntermediateStepType.LLM_NEW_TOKEN: IntermediateStepState.CHUNK,
            IntermediateStepType.TOOL_START: IntermediateStepState.START,
            IntermediateStepType.TOOL_END: IntermediateStepState.END,
            IntermediateStepType.WORKFLOW_START: IntermediateStepState.START,
            IntermediateStepType.WORKFLOW_END: IntermediateStepState.END,
            IntermediateStepType.TASK_START: IntermediateStepState.START,
            IntermediateStepType.TASK_END: IntermediateStepState.END,
            IntermediateStepType.FUNCTION_START: IntermediateStepState.START,
            IntermediateStepType.FUNCTION_END: IntermediateStepState.END,
            IntermediateStepType.CUSTOM_START: IntermediateStepState.START,
            IntermediateStepType.CUSTOM_END: IntermediateStepState.END,
            IntermediateStepType.SPAN_START: IntermediateStepState.START,
            IntermediateStepType.SPAN_CHUNK: IntermediateStepState.CHUNK,
            IntermediateStepType.SPAN_END: IntermediateStepState.END,
        }
        return mapping[self.event_type]

    @model_validator(mode="after")
    def validate_span_timestamp(self) -> "IntermediateStepPayload":
        if self.event_state != IntermediateStepState.END and self.span_event_timestamp is not None:
            raise ValueError("span_event_timestamp must be None unless event_state is END")
        return self


class IntermediateStep(BaseModel):
    model_config = ConfigDict(extra="forbid")
    parent_id: str
    function_ancestry: InvocationNode
    payload: IntermediateStepPayload

    @property
    def event_type(self) -> IntermediateStepType:
        return self.payload.event_type

    @property
    def event_timestamp(self) -> float:
        return self.payload.event_timestamp

    @property
    def span_event_timestamp(self) -> float | None:
        return self.payload.span_event_timestamp

    @property
    def framework(self) -> str | None:
        return self.payload.framework

    @property
    def name(self) -> str | None:
        return self.payload.name

    @property
    def tags(self) -> list[str] | None:
        return self.payload.tags

    @property
    def metadata(self) -> dict[str, typing.Any] | TraceMetadata | None:
        return self.payload.metadata

    @property
    def data(self) -> StreamEventData | None:
        return self.payload.data

    @property
    def usage_info(self) -> UsageInfo | None:
        return self.payload.usage_info

    @property
    def UUID(self) -> str:
        return self.payload.UUID

    @property
    def event_category(self) -> IntermediateStepCategory:
        return self.payload.event_category

    @property
    def event_state(self) -> IntermediateStepState:
        return self.payload.event_state
