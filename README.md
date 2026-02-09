# echo-agent

Echo back the user input using a single tool call.

## Quick Start

```bash
# Install dependencies
uv sync

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys (see ENV_SETUP.md for details)

# Run the agent
uv run python -m echo_agent
```

## Configuration

This agent uses **None** with model `databricks/databricks-claude-3-7-sonnet`.

Key settings (via environment or `config.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `databricks/databricks-claude-3-7-sonnet` | LLM model identifier |
| `PLANNER_MAX_ITERS` | `3` | Maximum planning iterations |
| `PLANNER_HOP_BUDGET` | `3` | Maximum tool hops |

See `ENV_SETUP.md` for complete environment setup instructions.

## Project Structure

```
echo_agent/
├── orchestrator.py     # Main entry point
├── planner.py          # ReactPlanner configuration
├── config.py           # Environment-driven settings
├── models.py           # Pydantic data models
├── tools/              # Agent tools
│   ├── echo.py
└── __main__.py         # CLI entry point
```

## Tools

This agent includes 1 tool(s):

### echo

Echo the provided message verbatim.


## Development

```bash
# Run tests
uv run pytest

# Type checking
uv run mypy src/

# Lint
uv run ruff check src/
```


## Customization

### Adding New Tools

1. Create a new tool file in `src/echo_agent/tools/`
2. Define input/output models and the async tool function
3. Register the tool in `tools/__init__.py`

### Adjusting Prompts

Pass `system_prompt_extra` to `ReactPlanner` in `planner.py`:

```python
planner = ReactPlanner(
    llm=config.llm_model,
    nodes=nodes,
    registry=registry,
    system_prompt_extra="Your custom instructions here...",
)
```

### Switching LLM Providers

Update your `.env` file with the appropriate API key and model:

```bash
# OpenAI
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o

# Anthropic
ANTHROPIC_API_KEY=sk-ant-...
LLM_MODEL=anthropic/claude-3-5-sonnet

# OpenRouter
OPENROUTER_API_KEY=sk-or-...
LLM_MODEL=openrouter/openai/gpt-4o
```

## Learn More

- [PenguiFlow Documentation](https://github.com/yourorg/penguiflow)
- [ReactPlanner Integration Guide](./REACT_PLANNER_INTEGRATION_GUIDE.md)