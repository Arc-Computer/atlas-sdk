# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapted from NeMo Agent Toolkit nat.agent.dual_node."""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from atlas.runtime.agent_loop.base_agent import AgentDecision
from atlas.runtime.agent_loop.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class DualNodeAgent(BaseAgent):
    def __init__(
        self,
        llm: BaseChatModel,
        tools: list[BaseTool],
        callbacks: list[AsyncCallbackHandler] | None = None,
        detailed_logs: bool = False,
        log_response_max_chars: int = 1000,
    ):
        super().__init__(
            llm=llm,
            tools=tools,
            callbacks=callbacks,
            detailed_logs=detailed_logs,
            log_response_max_chars=log_response_max_chars,
        )

    @abstractmethod
    async def agent_node(self, state: BaseModel) -> BaseModel:
        raise NotImplementedError

    @abstractmethod
    async def tool_node(self, state: BaseModel) -> BaseModel:
        raise NotImplementedError

    @abstractmethod
    async def conditional_edge(self, state: BaseModel) -> str:
        raise NotImplementedError

    async def _build_graph(self, state_schema: type) -> CompiledStateGraph:
        logger.debug("Building agent graph")
        graph = StateGraph(state_schema)
        graph.add_node("agent", self.agent_node)
        graph.add_node("tool", self.tool_node)
        graph.add_edge("tool", "agent")
        possible_edges = {AgentDecision.TOOL: "tool", AgentDecision.END: "__end__"}
        graph.add_conditional_edges("agent", self.conditional_edge, possible_edges)
        graph.set_entry_point("agent")
        self.graph = graph.compile()
        return self.graph

__all__ = ["DualNodeAgent"]
