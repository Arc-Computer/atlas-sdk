# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapted from NeMo Agent Toolkit nat.agent.base."""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC
from abc import abstractmethod
from enum import Enum
from typing import Any
from typing import Sequence

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

AGENT_LOG_PREFIX = "[AGENT]"
AGENT_CALL_LOG_MESSAGE = "%s\nAgent input: %s\nAgent output: %s"
TOOL_CALL_LOG_MESSAGE = "%s\nTool name: %s\nTool input: %s\nTool response: %s"


class AgentDecision(Enum):
    TOOL = "tool"
    END = "finished"


class BaseAgent(ABC):
    def __init__(
        self,
        llm: BaseChatModel,
        tools: Sequence[BaseTool],
        callbacks: Sequence[AsyncCallbackHandler] | None = None,
        detailed_logs: bool = False,
        log_response_max_chars: int = 1000,
    ) -> None:
        self.llm = llm
        self.tools = list(tools)
        self.callbacks = list(callbacks or [])
        self.detailed_logs = detailed_logs
        self.log_response_max_chars = log_response_max_chars
        self.graph = None
    @abstractmethod
    async def _build_graph(self, state_schema: type) -> Any:
        raise NotImplementedError
    async def _call_llm(self, llm: Runnable, inputs: dict[str, Any], config: RunnableConfig | None = None) -> AIMessage:
        response = await llm.ainvoke(inputs, config=config)
        content = getattr(response, "content", response)
        return AIMessage(content=str(content))
    async def _call_tool(
        self,
        tool: BaseTool,
        tool_input: dict[str, Any] | str,
        config: RunnableConfig | None = None,
        max_retries: int = 3,
    ) -> ToolMessage:
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                result = await tool.ainvoke(tool_input, config=config)
                if isinstance(result, dict):
                    result = [result]
                return ToolMessage(name=tool.name, tool_call_id=tool.name, content=result)
            except Exception as exc:
                last_error = exc
                if attempt == max_retries:
                    break
                await asyncio.sleep(2**attempt)
        message = f"Tool call failed: {last_error}"
        return ToolMessage(name=tool.name, tool_call_id=tool.name, content=message, status="error")
    def _log_tool_response(self, tool_name: str, tool_input: Any, tool_response: Any) -> None:
        if not self.detailed_logs:
            return
        response_text = str(tool_response)
        if len(response_text) > self.log_response_max_chars:
            response_text = response_text[: self.log_response_max_chars] + "...(truncated)"
        logger.info(TOOL_CALL_LOG_MESSAGE, AGENT_LOG_PREFIX, tool_name, tool_input, response_text)
    def _log_agent_decision(self, agent_input: Sequence[BaseMessage], agent_output: AIMessage) -> None:
        if not self.detailed_logs:
            return
        input_text = "\n".join(str(message.content) for message in agent_input)
        logger.info(AGENT_CALL_LOG_MESSAGE, AGENT_LOG_PREFIX, input_text, agent_output.content)

__all__ = [
    "AGENT_LOG_PREFIX",
    "AGENT_CALL_LOG_MESSAGE",
    "TOOL_CALL_LOG_MESSAGE",
    "AgentDecision",
    "BaseAgent",
]
