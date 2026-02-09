"""Integration tests for the EchoAgentOrchestrator."""

from __future__ import annotations

import pytest


from penguiflow.planner import PlannerFinish, PlannerPause

from echo_agent.config import Config
from echo_agent.orchestrator import EchoAgentOrchestrator



@pytest.fixture
def config() -> Config:
    return Config()





@pytest.mark.asyncio
async def test_execute_returns_agent_response(config: Config) -> None:
    orchestrator = EchoAgentOrchestrator(config)
    response = await orchestrator.execute(
        query="Tell me about PenguiFlow",
        tenant_id="tenant-a",
        user_id="user-a",
        session_id="session-a",
    )

    assert response.answer
    assert response.trace_id


    await orchestrator.stop()



@pytest.mark.asyncio
async def test_memory_disabled(config: Config) -> None:
    orchestrator = EchoAgentOrchestrator(config)
    response = await orchestrator.execute(
        query="memory off",
        tenant_id="tenant-b",
        user_id="user-b",
        session_id="session-b",
    )
    assert response.answer
    await orchestrator.stop()



@pytest.mark.asyncio
async def test_stop_marks_orchestrator_inactive(config: Config) -> None:
    orchestrator = EchoAgentOrchestrator(config)
    assert orchestrator._started is True  # type: ignore[attr-defined]
    await orchestrator.stop()
    assert orchestrator._started is False  # type: ignore[attr-defined]





