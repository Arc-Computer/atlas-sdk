# Example Configuration Files

This directory contains example Atlas configuration files demonstrating different agent adapter types.

## Naming Convention

### Agent Configuration Files

Complete runnable configurations follow the pattern: **`{adapter_type}_agent.yaml`**

- `{adapter_type}` - The type of agent adapter (e.g., `litellm`, `python`, `http_api`)
- `agent.yaml` - Indicates this is a complete agent configuration

**Examples:**
- `openai_agent.yaml` → Uses LiteLLM adapter (multi-provider support via litellm)
- `python_agent.yaml` → Uses Python adapter  
- `http_agent.yaml` → Uses HTTP API adapter

### Template Files

Template files (snippets showing partial configurations) use descriptive names prefixed with the section they document. Currently, no template files are present in this directory. See [Configuration Guide](../../docs/configs/configuration.md) for comprehensive configuration examples and reference.

## Available Examples

| File | Adapter Type | Description |
|------|--------------|-------------|
| `openai_agent.yaml` | `litellm` | Multi-provider agent (OpenAI, Anthropic, Gemini, etc.) with security review prompt |
| `http_agent.yaml` | `http_api` | HTTP-based agent calling external REST APIs |
| `python_agent.yaml` | `python` | Python adapter referencing the production-ready `mcp_tool_learning` example |

## Production-Ready Example

For a complete, production-ready example with progressive learning across 25 tasks, see [`examples/mcp_tool_learning/`](../../examples/mcp_tool_learning/README.md).

## Usage

Use these configs as starting points for your own Atlas deployments:

```bash
# Run with LiteLLM adapter (supports OpenAI, Anthropic, Gemini, Bedrock, X.AI)
atlas run --config configs/examples/openai_agent.yaml --task "Your task here"

# Run with HTTP adapter
atlas run --config configs/examples/http_agent.yaml --task "Your task here"

# Run with Python adapter (requires mcp_tool_learning setup)
atlas run --config configs/examples/python_agent.yaml --task "Your task here"
```

## Configuration Reference

For complete configuration documentation, see [Configuration Guide](../../docs/configs/configuration.md).

