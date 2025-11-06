# Supported Agent Patterns

Atlas autodiscovery supports multiple patterns for defining agents and environments. This guide helps you understand what works out-of-the-box and what requires workarounds.

## Fully Supported Patterns

### 1. Class-Based Agents

Agents defined as classes with `act()` or `invoke()` methods are fully supported:

```python
class MyAgent:
    def __init__(self, model: str = "gpt-5-mini"):
        self.model = model

    def act(self, observation: str) -> str:
        # Agent logic here
        return self.execute_task(observation)
```

**Autodiscovery:** Atlas will detect this class automatically and generate a factory.

**Usage:**
```bash
atlas env init  # Finds MyAgent automatically
atlas run --config .atlas/generated_config.yaml --task "Your task"
```

### 2. Factory Functions

Functions that return agent instances are fully supported:

```python
def create_agent(model: str = "gpt-5-mini", **kwargs):
    return create_deep_agent(
        model=model,
        tools=[internet_search, file_operations],
        **kwargs
    )
```

**Autodiscovery:** Atlas detects factory functions by name patterns (create_, build_, make_) and return type analysis.

**Usage:**
```bash
atlas env init  # Finds create_agent automatically
atlas run --config .atlas/generated_config.yaml --task "Your task"
```

### 3. Module-Level Instances (Limited Support)

Module-level agent instances work when no runtime configuration is needed:

```python
# src/agent/graph.py
from deepagents import create_deep_agent

agent = create_deep_agent(
    tools=[internet_search],
    instructions=research_instructions,
    subagents=[analysis_subagent],
)
```

**Autodiscovery:** Atlas detects these instances but requires explicit hints:

```bash
atlas env init --agent-fn src.agent.graph:agent
```

**Limitation:** Module-level instances cannot accept runtime kwargs. If you need configuration flexibility, convert to a factory function (see Workarounds below).

---

## Framework-Specific Patterns

### LangGraph CompiledStateGraph

```python
from langgraph.graph import StateGraph

graph = StateGraph(AgentState)
# ... add nodes and edges
compiled_app = graph.compile()
```

**Autodiscovery:** Supports compiled graphs. Use `--agent-fn` to point to the compiled instance.

### LangChain LCEL Runnable

```python
from langchain_core.runnables import RunnableSequence

chain = prompt | llm | output_parser
```

**Autodiscovery:** Supports LCEL chains as module-level instances.

### AutoGen ConversableAgent

```python
from autogen import ConversableAgent

agent = ConversableAgent(
    name="assistant",
    llm_config={"model": "gpt-5-mini"},
)
```

**Autodiscovery:** Supports as module-level instance or wrapped in factory.

---

## Workarounds

### Converting Module-Level Instance to Factory

If you have a module-level instance that needs runtime configuration:

**Before:**
```python
# src/agent/graph.py
agent = create_deep_agent(
    tools=[internet_search],
    instructions=research_instructions,
)
```

**After:**
```python
# src/agent/graph.py
def create_agent(**kwargs):
    """Factory function for Atlas runtime configuration."""
    return create_deep_agent(
        tools=kwargs.get('tools', [internet_search]),
        instructions=kwargs.get('instructions', research_instructions),
        **kwargs
    )

# Keep the original for direct use if needed
agent = create_agent()
```

**Then use:**
```bash
atlas env init --agent-fn src.agent.graph:create_agent
```

### Handling Import Errors

If discovery fails to import your agent module:

1. **Check PYTHONPATH:** Ensure your project root is discoverable:
   ```bash
   export PYTHONPATH=/path/to/your/project:$PYTHONPATH
   atlas env init
   ```

2. **Use explicit path:** Specify the agent module directly:
   ```bash
   atlas env init --agent-fn your_module.submodule:agent_name
   ```

3. **Review dependencies:** Ensure all required packages are installed:
   ```bash
   pip install -r requirements.txt
   ```

---

## Troubleshooting

### "No environment candidates detected"

**Cause:** Atlas couldn't find agent classes or factories automatically.

**Solutions:**
1. Use `--agent-fn` to specify the agent explicitly
2. Ensure agent has `act()` or `invoke()` methods
3. Check that agent files are in `.py` format (not notebooks)

### "Unsupported target – expected class or factory callable"

**Cause:** You're pointing to a module-level instance with runtime kwargs.

**Solution:** Convert to factory function (see Workarounds above).

### Discovery times out or fails

**Cause:** Synthesis hitting token limits or timeouts.

**Solutions:**
1. Update to latest Atlas SDK (includes increased limits)
2. Use `--agent-fn` to skip full autodiscovery
3. Check `.atlas/discover.json` for partial results

---

## Manual Configuration

When autodiscovery isn't suitable, you can write configuration manually:

```yaml
# config.yaml
agent:
  type: python
  import_path: your_module.agent
  attribute: agent_instance  # or factory_function
  working_directory: ./
  allow_generator: false
  llm:
    provider: openai
    model: gpt-5-mini
    api_key_env: OPENAI_API_KEY
```

Then run:
```bash
atlas run --config config.yaml --task "Your task"
```

---

## See Also

- [Quickstart Guide](quickstart.mdx) - Get started with Atlas SDK
- [Configuration Reference](../configs/configuration.md) - Full config options
- [GitHub Issues](https://github.com/Arc-Computer/atlas-sdk/issues) - Report discovery issues
