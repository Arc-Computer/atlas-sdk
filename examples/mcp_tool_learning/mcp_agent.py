"""MCP-Powered Agent for File Operations Learning

This agent uses MCP tools via langchain-mcp-adapters to perform file operations.
It demonstrates how agents can learn to effectively use MCP tools through Atlas SDK.
"""

import os
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


async def create_agent(prompt=None, metadata=None):
    """Create an agent that uses MCP tools for file operations.

    This agent connects to a local MCP server that provides file system tools.
    It uses LangGraph's ReAct pattern to reason about and execute tool calls.

    Args:
        prompt: Optional task prompt for runtime invocation
        metadata: Optional metadata for runtime invocation

    Returns:
        A LangGraph ReAct agent with MCP tools (factory mode) or
        agent execution result (runtime mode)
    """
    # Check if this is a runtime invocation (has prompt) or factory invocation
    is_runtime = prompt is not None
    # Get the absolute path to the MCP server script
    server_path = Path(__file__).parent / "mcp_server.py"

    # Configure MCP client to connect to our file operations server
    client = MultiServerMCPClient(
        {
            "file-operations": {
                "command": sys.executable,  # Use the current Python interpreter
                "args": [str(server_path)],
                "transport": "stdio",
            }
        }
    )

    # Get tools from the MCP server
    tools = await client.get_tools()

    # Initialize the LLM
    # Use GPT-4.1-mini for cost efficiency (~$0.15 per million tokens input)
    llm = ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.1,  # Low temperature for consistent, focused behavior
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Create a ReAct agent with the MCP tools
    agent = create_react_agent(
        llm,
        tools,
        prompt=(
            "You are a file operations assistant. Use the available tools to help with file management tasks. "
            "Always use the most appropriate tool for each task. Be efficient and avoid redundant operations."
        ),
    )

    # If runtime invocation, execute the agent and return the result
    if is_runtime:
        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )
        # Extract the final message content
        messages = response.get("messages", [])
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, "content"):
                return last_message.content
            return str(last_message)
        return str(response)

    # Otherwise return the agent instance for factory mode
    return agent


def create_sync_agent():
    """Synchronous wrapper for agent creation (for Atlas compatibility).

    Atlas SDK supports both sync and async agent factories. This function
    provides a sync interface that returns the async agent.
    """
    import asyncio

    # Check if we're already in an event loop
    try:
        loop = asyncio.get_running_loop()
        # We're in an async context, just create the agent
        return asyncio.create_task(create_agent())
    except RuntimeError:
        # No event loop, create one
        return asyncio.run(create_agent())


if __name__ == "__main__":
    # Test the agent creation
    import asyncio

    async def test():
        agent = await create_agent()

        # Test with a simple query
        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "List files in the current directory"}]}
        )

        print("Agent response:")
        for message in response["messages"]:
            if hasattr(message, "content"):
                print(f"{message.type}: {message.content}")

    asyncio.run(test())
