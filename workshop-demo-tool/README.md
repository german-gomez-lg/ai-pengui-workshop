# Workshop Demo Tool: mock_campaign_insights

This folder contains a ready-to-drop tool implementation and a small fixture dataset.

## Files
- `tools/mock_campaign_insights.py`: Tool implementation
- `data/mock_campaigns.json`: Fixture dataset

## How to wire it
1. Add this tool to `echo-agent.yaml` under `tools:`

```yaml
  - name: mock_campaign_insights
    description: "Return mock campaign insights from a local fixture."
    side_effects: read
    args:
      campaign_id: Optional[str]
      client_name: Optional[str]
      status: Optional[str]
      date_range: Optional[str]
      metrics: Optional[list[str]]
      include_recommendations: Optional[bool]
      include_campaigns: Optional[bool]
    result:
      client_name: Optional[str]
      campaign_id: Optional[str]
      date_range: Optional[str]
      metrics: dict[str, float]
      campaigns: list[dict[str, any]]
      recommendations: list[str]
      notes: list[str]
```

2. Update `planner.system_prompt_extra` in `echo-agent.yaml` to allow tool use.
   Replace the current echo-only rules with something like:

```yaml
  system_prompt_extra: |
    You are a demo ads insights agent.

    Rules:
    - If the user asks for campaign performance or insights, call `mock_campaign_insights`.
    - Otherwise, ask a short clarifying question or answer directly.
```

3. Regenerate the agent files:

```bash
penguiflow generate --spec echo-agent.yaml --force
```

4. Copy the tool implementation and fixture:

```bash
cp workshop-demo-tool/tools/mock_campaign_insights.py src/echo_agent/tools/mock_campaign_insights.py
mkdir -p data
cp workshop-demo-tool/data/mock_campaigns.json data/mock_campaigns.json
```

5. Run a quick local test:

```bash
uv run python -m echo_agent "Do we have active campaigns from Ford? How are they doing?"
```

```bash
uv run python -m echo_agent "Show me the campaigns from Disney and add recommendations to improve them"
```

## Notes
- The tool looks for `data/mock_campaigns.json` first. If it is missing, it will fallback to `workshop-demo-tool/data/mock_campaigns.json`.
- You can also set `MOCK_CAMPAIGN_DATA_PATH` to an absolute path in Databricks or other deployments.
- Keep the fixture small and deterministic for a smooth demo.
