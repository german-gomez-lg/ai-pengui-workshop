# Environment Setup for echo-agent

This document explains how to configure environment variables for your agent.

## Quick Start

1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

2. Set your LLM provider API key (see Your Configuration below)

3. Run your agent:
   ```bash
   uv run echo_agent
   # or use the playground:
   penguiflow dev .
   ```

## Your Agent Configuration

Your agent is configured to use:
- **Model**: `databricks/databricks-claude-3-7-sonnet`

### OpenAI Setup (Your Configured Provider)

```bash
OPENAI_API_KEY=sk-proj-...
LLM_MODEL=databricks/databricks-claude-3-7-sonnet
```

Get your API key at: https://platform.openai.com/api-keys


---

## Other Providers (Reference)

PenguiFlow uses [LiteLLM](https://docs.litellm.ai/) for LLM integration. You can switch providers by changing your `.env`:

<details>
<summary>OpenAI</summary>

```bash
OPENAI_API_KEY=sk-proj-...
LLM_MODEL=gpt-4o
```
</details>

<details>
<summary>Anthropic (Claude)</summary>

```bash
ANTHROPIC_API_KEY=sk-ant-api03-...
LLM_MODEL=anthropic/claude-sonnet-4-20250514
```
</details>

<details>
<summary>OpenRouter</summary>

```bash
OPENROUTER_API_KEY=sk-or-v1-...
LLM_MODEL=openrouter/anthropic/claude-sonnet-4-20250514
```
</details>

<details>
<summary>Azure OpenAI</summary>

```bash
AZURE_API_KEY=...
AZURE_API_BASE=https://your-resource.openai.azure.com/
AZURE_API_VERSION=2024-02-15-preview
LLM_MODEL=azure/your-deployment-name
```
</details>

<details>
<summary>Google (Gemini)</summary>

```bash
GEMINI_API_KEY=...
LLM_MODEL=gemini/gemini-1.5-pro
```
</details>

<details>
<summary>AWS Bedrock</summary>

```bash
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION_NAME=us-east-1
LLM_MODEL=bedrock/anthropic.claude-3-sonnet-20240229-v1:0
```
</details>

## Environment Variables Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_MODEL` | Primary LLM model identifier | `databricks/databricks-claude-3-7-sonnet` |
| `SUMMARIZER_MODEL` | Model for trajectory summarization (optional) | Same as LLM_MODEL |
| `REFLECTION_MODEL` | Model for reflection/critique (optional) | Same as LLM_MODEL |
| `MEMORY_ENABLED` | Enable memory service integration | `false` |
| `SUMMARIZER_ENABLED` | Enable trajectory summarization | `false` |
| `REFLECTION_ENABLED` | Enable reflection loop | `false` |
| `SHORT_TERM_MEMORY_ENABLED` | Enable built-in short-term memory | `false` |
| `MEMORY_BASE_URL` | Memory service endpoint | `http://localhost:8000` |
| `RAG_SERVER_BASE_URL` | RAG server endpoint | `http://localhost:8081` |
| `WAYFINDER_BASE_URL` | Wayfinder service endpoint | `http://localhost:8082` |
| `PLANNER_MAX_ITERS` | Maximum planner iterations | `3` |
| `PLANNER_HOP_BUDGET` | Maximum tool hops per run | `3` |
| `PLANNER_ABSOLUTE_MAX_PARALLEL` | Maximum parallel tool calls | `1` |
| `PLANNER_STREAM_FINAL_RESPONSE` | Stream final LLM answer tokens (true/false) | `False` |
| `PLANNER_MULTI_ACTION_SEQUENTIAL` | Execute extra tool actions when model emits multiple JSON objects (true/false) | `False` |
| `PLANNER_MULTI_ACTION_READ_ONLY_ONLY` | Only auto-execute extra actions for pure/read tools (true/false) | `True` |
| `PLANNER_MULTI_ACTION_MAX_TOOLS` | Max extra tool calls to auto-execute per LLM response | `2` |

Tuning options for built-in short-term memory are available as `SHORT_TERM_MEMORY_*` variables in `.env.example`.

## Security Notes

- **Never commit `.env` to version control** - it's already in `.gitignore`
- Use `.env.example` as a template (safe to commit)
- For production, use secret management (AWS Secrets Manager, Vault, etc.)
- Rotate API keys regularly