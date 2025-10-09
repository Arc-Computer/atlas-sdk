# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapted from NeMo Agent Toolkit nat.agent.tool_calling_agent.agent."""

from __future__ import annotations

import logging
from typing import Any
from typing import List

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel
from pydantic import Field

from atlas.runtime.agent_loop.dual_node import DualNodeAgent
from atlas.connectors.langchain_bridge import BYOABridgeLLM
from atlas.runtime.agent_loop.base_agent import AGENT_CALL_LOG_MESSAGE
from atlas.runtime.agent_loop.base_agent import AGENT_LOG_PREFIX
from atlas.runtime.agent_loop.base_agent import AgentDecision

logger = logging.getLogger(__name__)


class ToolCallAgentGraphState(BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)


class ToolCallAgentGraph(DualNodeAgent):
    def __init__(
        self,
        llm: BYOABridgeLLM,
        tools: List[BaseTool],
        system_prompt: str | None = None,
        callbacks: List[AsyncCallbackHandler] | None = None,
        detailed_logs: bool = False,
        log_response_max_chars: int = 1000,
        handle_tool_errors: bool = True,
        return_direct: List[BaseTool] | None = None,
    ):
        bound_llm = llm.bind_tools(tools)
        super().__init__(
            llm=bound_llm,
            tools=tools,
            callbacks=callbacks,
            detailed_logs=detailed_logs,
            log_response_max_chars=log_response_max_chars,
        )
        if system_prompt:
            system_message = RunnableLambda(lambda state: [{"role": "system", "content": system_prompt}] + state.get("messages", []))
        else:
            system_message = RunnableLambda(lambda state: state.get("messages", []))
        self.agent = system_message | self.llm
        self.tool_caller = ToolNode(tools, handle_tool_errors=handle_tool_errors)
        self.return_direct = [tool.name for tool in return_direct] if return_direct else []

    async def agent_node(self, state: ToolCallAgentGraphState) -> ToolCallAgentGraphState:
        if not state.messages:
            raise RuntimeError("Agent node received empty message list")
        response = await self.agent.ainvoke({"messages": state.messages}, config=RunnableConfig(callbacks=self.callbacks))
        if self.detailed_logs:
            self._log_agent_decision(state.messages, response)
        state.messages.append(response)
        return state

    async def conditional_edge(self, state: ToolCallAgentGraphState) -> str:
        last_message = state.messages[-1]
        if last_message.tool_calls:
            return AgentDecision.TOOL
        return AgentDecision.END

    async def tool_node(self, state: ToolCallAgentGraphState) -> ToolCallAgentGraphState:
        tool_calls = state.messages[-1].tool_calls
        tool_input = state.messages[-1]
        result = await self.tool_caller.ainvoke(
            input={"messages": [tool_input]},
            config=RunnableConfig(callbacks=self.callbacks, configurable={}),
        )
        for message in result.get("messages", []):
            if self.detailed_logs:
                self._log_tool_response(str(tool_calls), tool_input, message.content)
            state.messages.append(message)
        return state

    async def tool_conditional_edge(self, state: ToolCallAgentGraphState) -> AgentDecision:
        if not state.messages:
            return AgentDecision.TOOL
        if self.return_direct:
            for message in reversed(state.messages):
                if isinstance(message, AIMessage) and message.tool_calls:
                    for call in message.tool_calls:
                        if call.name in self.return_direct:
                            return AgentDecision.END
                    break
        return AgentDecision.TOOL

    async def _build_graph(self, state_schema: type) -> CompiledStateGraph:
        graph = StateGraph(state_schema)
        graph.add_node("agent", self.agent_node)
        graph.add_node("tool", self.tool_node)
        if self.return_direct:
            graph.add_conditional_edges("tool", self.tool_conditional_edge, {AgentDecision.END: "__end__", AgentDecision.TOOL: "agent"})
        else:
            graph.add_edge("tool", "agent")
        graph.add_conditional_edges("agent", self.conditional_edge, {AgentDecision.TOOL: "tool", AgentDecision.END: "__end__"})
        graph.set_entry_point("agent")
        self.graph = graph.compile()
        return self.graph

    async def build_graph(self) -> CompiledStateGraph:
        if self.graph is None:
            await self._build_graph(ToolCallAgentGraphState)
        return self.graph

__all__ = ["ToolCallAgentGraph", "ToolCallAgentGraphState"]
