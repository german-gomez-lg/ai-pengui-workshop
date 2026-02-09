"""Tool: Return mock campaign insights from a local fixture."""

from __future__ import annotations

import json
from pathlib import Path
import os
from typing import Any

from pydantic import BaseModel, Field

from penguiflow.catalog import tool
from penguiflow.planner import ToolContext


class MockCampaignInsightsArgs(BaseModel):
    """Input for mock campaign insights."""

    campaign_id: str | None = Field(
        None, description="Campaign ID (e.g., CMP-1001). Optional."
    )
    client_name: str | None = Field(
        None, description="Client name (e.g., Ford, Disney). Optional."
    )
    status: str | None = Field(
        "active", description="Campaign status filter (active, paused, any)"
    )
    date_range: str | None = Field(
        "last_7d",
        description="Reporting window (e.g., last_7d, last_30d, 2026-01-01..2026-01-31)",
    )
    metrics: list[str] | None = Field(
        None, description="Optional list of metric keys to return"
    )
    include_recommendations: bool = Field(
        True, description="Include basic optimization recommendations"
    )
    include_campaigns: bool = Field(
        True, description="Include per-campaign details in the response"
    )


class MockCampaignInsightsResult(BaseModel):
    """Output for mock campaign insights."""

    client_name: str | None
    campaign_id: str | None
    date_range: str | None
    metrics: dict[str, float]
    campaigns: list[dict[str, Any]]
    recommendations: list[str]
    notes: list[str]


def _load_fixture() -> dict[str, Any]:
    env_path = os.getenv("MOCK_CAMPAIGN_DATA_PATH")
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))

    here = Path(__file__).resolve()
    candidates.extend(
        [
            here.parent / "mock_campaigns.json",
            here.parents[2] / "data" / "mock_campaigns.json",
            here.parents[3] / "data" / "mock_campaigns.json",
            here.parents[4] / "data" / "mock_campaigns.json",
            here.parents[4] / "src" / "echo_agent" / "data" / "mock_campaigns.json",
            here.parents[4] / "workshop-demo-tool" / "data" / "mock_campaigns.json",
            Path("/app/python/source_code/data/mock_campaigns.json"),
            Path("/app/python/source_code/src/echo_agent/data/mock_campaigns.json"),
            Path("/app/python/source_code/workshop-demo-tool/data/mock_campaigns.json"),
        ]
    )

    for path in candidates:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))

    raise FileNotFoundError(
        "mock_campaigns.json not found. "
        "Set MOCK_CAMPAIGN_DATA_PATH or place the file in ./data/."
    )


def _range_multiplier(date_range: str | None) -> float:
    if not date_range:
        return 1.0
    key = date_range.lower()
    if "last_30" in key:
        return 4.1
    if "last_90" in key:
        return 12.5
    if "yesterday" in key:
        return 0.14
    return 1.0


def _compute_metrics(base: dict[str, float], multiplier: float) -> dict[str, float]:
    impressions = base["impressions"] * multiplier
    clicks = base["clicks"] * multiplier
    spend = base["spend"] * multiplier
    conversions = base["conversions"] * multiplier

    ctr = (clicks / impressions) * 100 if impressions else 0.0
    cpm = (spend / impressions) * 1000 if impressions else 0.0
    cpa = (spend / conversions) if conversions else 0.0

    return {
        "impressions": round(impressions, 0),
        "clicks": round(clicks, 0),
        "spend": round(spend, 2),
        "conversions": round(conversions, 0),
        "ctr": round(ctr, 2),
        "cpm": round(cpm, 2),
        "cpa": round(cpa, 2),
    }


def _default_recommendations(metrics: dict[str, float]) -> list[str]:
    recs: list[str] = []
    if metrics.get("ctr", 0.0) < 0.35:
        recs.append("Refresh creatives; CTR is below 0.35%.")
    if metrics.get("cpa", 0.0) > 35:
        recs.append("Tighten targeting or adjust bids to reduce CPA.")
    if metrics.get("cpm", 0.0) > 18:
        recs.append("Test lower-floor inventory or adjust pacing to reduce CPM.")
    if not recs:
        recs.append("Performance is stable; continue current optimization cadence.")
    return recs


@tool(
    desc="Return mock campaign insights from a local fixture.",
    tags=["demo", "ads"],
    side_effects="read",
)
async def mock_campaign_insights(
    args: MockCampaignInsightsArgs, ctx: ToolContext
) -> MockCampaignInsightsResult:
    if hasattr(ctx, "emit_chunk"):
        await ctx.emit_chunk("Loading mock campaign data...")
    if args.client_name:
        ctx.tool_context["last_client"] = args.client_name

    data = _load_fixture()
    campaigns = data.get("campaigns", [])

    notes: list[str] = []

    filtered = campaigns
    if args.client_name:
        filtered = [
            c
            for c in filtered
            if c.get("client", "").lower() == args.client_name.lower()
        ]
    if args.campaign_id:
        filtered = [
            c for c in filtered if c.get("campaign_id") == args.campaign_id
        ]
    if args.status and args.status.lower() != "any":
        filtered = [
            c
            for c in filtered
            if c.get("status", "").lower() == args.status.lower()
        ]

    if not filtered:
        if hasattr(ctx, "emit_chunk"):
            await ctx.emit_chunk("No matching campaigns found.")
        notes.append("No matching campaigns found; returning empty result set.")
        return MockCampaignInsightsResult(
            client_name=args.client_name,
            campaign_id=args.campaign_id,
            date_range=args.date_range,
            metrics={},
            campaigns=[],
            recommendations=[],
            notes=notes,
        )

    multiplier = _range_multiplier(args.date_range)

    campaign_rows: list[dict[str, Any]] = []
    totals = {"impressions": 0.0, "clicks": 0.0, "spend": 0.0, "conversions": 0.0}
    for c in filtered:
        m = _compute_metrics(c["base_metrics"], multiplier)
        for k in totals:
            totals[k] += m.get(k, 0.0)
        if args.metrics:
            m = {k: v for k, v in m.items() if k in set(args.metrics)}
        campaign_rows.append(
            {
                "campaign_id": c.get("campaign_id"),
                "name": c.get("name"),
                "client": c.get("client"),
                "status": c.get("status"),
                "metrics": m,
            }
        )

    overall = _compute_metrics(totals, 1.0)
    if args.metrics:
        overall = {k: v for k, v in overall.items() if k in set(args.metrics)}

    recommendations = (
        _default_recommendations(overall) if args.include_recommendations else []
    )

    if hasattr(ctx, "emit_chunk"):
        await ctx.emit_chunk("Computed metrics and recommendations.")

    return MockCampaignInsightsResult(
        client_name=args.client_name,
        campaign_id=args.campaign_id,
        date_range=args.date_range,
        metrics=overall,
        campaigns=campaign_rows if args.include_campaigns else [],
        recommendations=recommendations,
        notes=notes,
    )
