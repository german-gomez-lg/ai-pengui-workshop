# AI Assistant Instructions

> Context for AI coding assistants (Claude Code, Cursor, Copilot, Cody, etc.)

This project uses **PenguiFlow v2.7** for agent scaffolding and orchestration.

---

## Project Overview

- **Agent**: echo-agent
- **Spec File**: `echo-agent.yaml`
- **Package**: `echo_agent`
- **Template**: Will be set when you run `penguiflow generate`

---

## Workflow

### 1. Edit the Spec

The `echo-agent.yaml` file defines:
- Agent configuration (name, template, flags)
- LLM settings (provider, model)
- Tools (functions the agent can call)
- Flows (optional multi-step pipelines)
- Planner settings (prompts, limits)

### 2. Generate the Project

```bash
# Validate first
penguiflow generate --spec echo-agent.yaml --dry-run

# Generate
penguiflow generate --spec echo-agent.yaml

# If regenerating, use --force to overwrite
penguiflow generate --spec echo-agent.yaml --force
```

### 3. Implement Tools

Generated tools are stubs. Find them in `src/echo_agent/tools/`.

---

## Key Conventions

### Tool Names
- Must be `snake_case`: `search_documents`, not `searchDocuments`
- Avoid Python reserved words

### Type Annotations
Supported types in spec:
- `str`, `int`, `float`, `bool`
- `list[T]` (e.g., `list[str]`)
- `dict[K, V]` (e.g., `dict[str, int]`)
- `Optional[T]` (e.g., `Optional[int]`)

### Tool Implementation Pattern

```python
from penguiflow.catalog import tool
from penguiflow.planner import ToolContext

@tool(desc="Description shown to LLM", tags=["tag"], side_effects="read")
async def my_tool(args: MyToolArgs, ctx: ToolContext) -> MyToolResult:
    # Access context
    llm_data = ctx.llm_context.get("key")      # Read-only LLM context
    runtime_data = ctx.tool_context.get("key")  # Mutable runtime context

    # Implementation
    result = await do_something(args.param)

    return MyToolResult(output=result)
```

### Side Effects

| Value | Meaning |
|-------|---------|
| `pure` | No side effects, deterministic |
| `read` | Reads external state |
| `write` | Writes/mutates external state |
| `external` | Calls external APIs |
| `stateful` | Maintains internal state |

---

## Common Tasks

### Add a New Tool

1. Add to `echo-agent.yaml`:
```yaml
tools:
  - name: new_tool
    description: "What this tool does"
    args:
      input_param: str
    result:
      output_field: str
```

2. Regenerate: `penguiflow generate --spec echo-agent.yaml --force`

3. Implement in `src/echo_agent/tools/new_tool.py`

### Add a Flow (Pipeline)

1. Add to `echo-agent.yaml`:
```yaml
flows:
  - name: my_pipeline
    description: "What this pipeline does"
    nodes:
      - name: step_one
        description: "First step"
      - name: step_two
        description: "Second step"
    steps: [step_one, step_two]
```

2. Regenerate with `--force`

3. Flow and orchestrator generated in `src/echo_agent/flows/`

### Modify System Prompt

Edit `planner.system_prompt_extra` in `echo-agent.yaml`, then regenerate.

---

## File Locations

| File | Purpose |
|------|---------|
| `echo-agent.yaml` | Agent specification |
| `src/echo_agent/config.py` | Environment config |
| `src/echo_agent/orchestrator.py` | Main entry point |
| `src/echo_agent/planner.py` | Tool catalog, prompts |
| `src/echo_agent/tools/*.py` | Tool implementations |
| `src/echo_agent/flows/*.py` | Flow definitions |
| `.env` | Environment variables (API keys) |

---

## Commands

```bash
# Generate from spec
penguiflow generate --spec echo-agent.yaml

# Validate spec without generating
penguiflow generate --spec echo-agent.yaml --dry-run

# Force regenerate (overwrites files)
penguiflow generate --spec echo-agent.yaml --force

# Run the agent
uv run python -m echo_agent "Your query"

# Run tests
uv run pytest
```

---

## Don't Modify (Regenerated Files)

These files are overwritten on regenerate:
- `src/echo_agent/tools/__init__.py`
- `src/echo_agent/planner.py`
- `src/echo_agent/flows/__init__.py`
- `src/echo_agent/flows/*.py`
- `src/echo_agent/config.py`

**Safe to modify:**
- Individual tool files (`tools/*.py` except `__init__.py`)
- `orchestrator.py` (use `--force` carefully)
- Test files
- `.env`

---

## Debugging

### Verbose Generation
```bash
penguiflow generate --spec echo-agent.yaml --verbose
```

### Check Generated Code Syntax
```bash
python -m py_compile src/echo_agent/tools/*.py
```

### Run Single Tool Test
```bash
uv run pytest tests/test_tools/test_my_tool.py -v
```