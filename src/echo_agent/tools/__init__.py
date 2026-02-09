"""Tool registry for echo-agent."""

from __future__ import annotations

from penguiflow.node import Node
from penguiflow.registry import ModelRegistry

from .echo import EchoArgs, EchoResult, echo

__all__ = [
    "echo",
    "EchoArgs",
    "EchoResult",
    "build_catalog_bundle",
]


def build_catalog_bundle() -> tuple[list[Node], ModelRegistry]:
    """Create the planner catalog and registry."""
    registry = ModelRegistry()
    registry.register("echo", EchoArgs, EchoResult)
    nodes = [
        Node(echo, name="echo"),
    ]
    return nodes, registry
