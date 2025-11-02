# Evaluation Configs

This directory contains configuration files for running evaluations and experiments. Configs are organized by evaluation type.

## Directory Structure

```
configs/eval/
├── reward/
│   └── models.yaml          # Reward judge model presets and combos
└── learning/
    ├── baseline_openai.yaml       # Baseline learning config (OpenAI)
    ├── baseline_claude.yaml       # Baseline learning config (Claude)
    ├── scope_shift_openai.yaml    # Scope shift experiment (OpenAI)
    ├── tool_adoption_openai.yaml  # Tool adoption experiment (OpenAI)
    └── tool_adoption_claude.yaml   # Tool adoption experiment (Claude)
```

## Reward Configs (`reward/`)

### `models.yaml`

Defines judge model presets and judge combo pairings for reward system evaluation.

**Structure:**
- `presets`: Individual judge model configurations (provider, model, API keys, etc.)
- `combos`: Judge pairings (small judge → large judge/arbiter)

**Usage:**
Used by `scripts/benchmark_reward_models.py` to evaluate different judge combinations. The script loads this file automatically and falls back to built-in defaults if missing.

**Example:**
```yaml
presets:
  gemini-2.5-flash:
    provider: GEMINI
    model: gemini/gemini-2.5-flash
    # ...

combos:
  gemini_pair:
    small_preset: gemini-2.5-flash
    large_preset: gemini-2.5-pro
```

## Learning Configs (`learning/`)

Learning experiment configs follow the naming pattern: `<experiment>_<provider>.yaml`

### Baseline Configs

- **`baseline_openai.yaml`** - Baseline learning configuration using OpenAI/Gemini models
  - Student: GPT-5 Mini
  - Teacher: GPT-5 Mini
  - Synthesiser: Gemini 2.5 Flash
  - Focus: Reinforcement-focused prompt

- **`baseline_claude.yaml`** - Baseline learning configuration using Claude models
  - Student: Claude Haiku 4.5
  - Teacher: Claude Sonnet 4.5
  - Synthesiser: Claude Sonnet 4.5
  - Focus: Reinforcement-focused prompt with impact instrumentation

### Experiment Configs

- **`scope_shift_openai.yaml`** - Scope shift experiment
  - Focus: Differentiation and transfer hypotheses
  - Default scope category: `differentiation`
  - Tests cross-incident generalization

- **`tool_adoption_openai.yaml`** - Tool adoption experiment with OpenAI
  - Includes tool definitions (web_search, calculate, format_text)
  - Tests learning with tool-backed adapters
  - Usage tracking enabled

- **`tool_adoption_claude.yaml`** - Tool adoption experiment with Claude
  - Same tools as OpenAI version
  - Uses Claude models for student/teacher/synthesiser
  - Tests provider-specific tool adoption behavior

## Usage

### Running Learning Experiments

```bash
# Run baseline experiment
atlas run --config configs/eval/learning/baseline_openai.yaml --task "Your task"

# Run tool adoption experiment
atlas run --config configs/eval/learning/tool_adoption_claude.yaml --task "Search and calculate"
```

### Running Reward Evaluations

```bash
# Benchmark reward models (uses configs/eval/reward/models.yaml)
python -m scripts.benchmark_reward_models \
  --dataset atlas/data/reward_eval_trajectories.jsonl
```

## Naming Convention

- **Learning configs**: `<experiment>_<provider>.yaml`
  - Examples: `baseline_openai.yaml`, `tool_adoption_claude.yaml`
  - Makes it clear what experiment and which provider the config targets

- **Reward configs**: Descriptive names like `models.yaml`
  - Single config file for judge presets/combos

## Creating New Configs

When adding new learning experiment configs:

1. Follow the naming pattern: `<experiment>_<provider>.yaml`
2. Include clear metadata with experiment name and learning_key
3. Document any experiment-specific settings in the config comments
4. Update this README if adding a new experiment category

Example:
```yaml
metadata:
  experiment:
    name: 20251030_my_experiment_openai
    learning_key: runtime.synthetic.my_experiment
    seed: 20251030
```

