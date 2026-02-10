"""Tests for echo tool."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from echo_agent.tools import echo, EchoArgs

if TYPE_CHECKING:
    from conftest import DummyToolContext


@pytest.mark.asyncio
async def test_echo_not_implemented(dummy_ctx: "DummyToolContext") -> None:
    """Tool stub raises NotImplementedError until implemented.

    TODO: Replace this test with actual behavior tests once implemented.
    """
    args = EchoArgs(message="example")

    with pytest.raises(NotImplementedError, match="echo"):
        await echo(args, dummy_ctx)
