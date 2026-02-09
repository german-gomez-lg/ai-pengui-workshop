"""Tool: Echo the provided message verbatim."""

from __future__ import annotations

from pydantic import BaseModel

from penguiflow.catalog import tool
from penguiflow.planner import ToolContext


class EchoArgs(BaseModel):
    """Echo the provided message verbatim. input."""
    message: str


class EchoResult(BaseModel):
    """Echo the provided message verbatim. output."""
    message: str


@tool(
    desc="Echo the provided message verbatim.",
    tags=[],
    side_effects="pure",
)
async def echo(args: EchoArgs, ctx: ToolContext) -> EchoResult:
    """Echo the provided message verbatim. """
    del ctx  # Remove when implementing - ctx provides emit_chunk(), pause(), etc.
    return EchoResult(message=args.message)