"""Tool registry for echo-agent."""

from __future__ import annotations

from penguiflow.node import Node
from penguiflow.registry import ModelRegistry

from .echo import EchoArgs, EchoResult, echo
from .campaign_line import CampaignLineStatusArgs, CampaignLineStatusResult, check_campaign_line_status
from .search_segment import SegmentSearchArgs, SegmentSearchResult, search_segment

__all__ = [
    "echo",
    "EchoArgs",
    "EchoResult",
    "check_campaign_line_status",
    "CampaignLineStatusArgs",
    "CampaignLineStatusResult",
    "search_segment",
    "SegmentSearchArgs",
    "SegmentSearchResult",
    "build_catalog_bundle",
]


def build_catalog_bundle() -> tuple[list[Node], ModelRegistry]:
    """Create the planner catalog and registry."""
    registry = ModelRegistry()
    registry.register("echo", EchoArgs, EchoResult)
    registry.register("check_campaign_line_status", CampaignLineStatusArgs, CampaignLineStatusResult)
    registry.register("search_segment", SegmentSearchArgs, SegmentSearchResult)

    nodes = [
        Node(echo, name="echo"),
        Node(check_campaign_line_status, name="check_campaign_line_status"),
        Node(search_segment, name="search_segment"),
    ]
    return nodes, registry