"""Main orchestrator for echo-agent."""

from __future__ import annotations

import logging
import secrets
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from penguiflow.errors import FlowError
from penguiflow.planner import PlannerFinish, PlannerPause



from penguiflow.steering import SteeringInbox

from .config import Config
from .planner import PlannerBundle, build_planner
from .telemetry import AgentTelemetry

_LOGGER = logging.getLogger(__name__)


class EchoAgentFlowError(RuntimeError):
    """Raised when planner execution fails."""

    def __init__(self, flow_error: FlowError | str) -> None:
        message = flow_error.message if isinstance(flow_error, FlowError) else str(flow_error)
        super().__init__(message)
        self.flow_error = flow_error


@dataclass
class AgentResponse:
    """Response envelope returned by the orchestrator."""

    answer: str | None
    trace_id: str
    metadata: dict[str, Any] | None = None




def _collect_streams(metadata: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Aggregate stream chunks from planner metadata."""
    streams: dict[str, list[dict[str, Any]]] = {}
    for step in metadata.get("steps", []):
        for stream_id, chunks in (step.get("streams") or {}).items():
            streams.setdefault(stream_id, []).extend(chunks)
    return streams


def _extract_answer(payload: Any) -> str | None:
    """Normalise planner payloads to a displayable answer."""

    if payload is None:
        return None
    if isinstance(payload, Mapping):
        for key in ("raw_answer", "answer", "text", "content", "message", "greeting", "response", "result"):
            if key in payload:
                value = payload.get(key)
                return None if value is None else str(value)
        return str(payload)

    for attr in ("raw_answer", "answer", "text", "content", "message", "greeting", "response", "result"):
        if hasattr(payload, attr):
            value = getattr(payload, attr)
            return None if value is None else str(value)

    return str(payload)


class EchoAgentOrchestrator:
    """Production-style orchestrator using ReactPlanner."""

    def __init__(
        self,
        config: Config,
        *,
        telemetry: AgentTelemetry | None = None,

    ) -> None:
        self._config = config

        self._memory = None

        self._telemetry = telemetry or AgentTelemetry(
            flow_name="echo-agent",
            logger=_LOGGER,
        )

        planner_bundle: PlannerBundle = build_planner(
            config,
            event_callback=self._telemetry.record_planner_event,
        )
        self._planner = planner_bundle.planner
        self._tool_context_defaults: dict[str, Any] = {}


        self._started = True



    async def execute(
        self,
        query: str,
        *,
        tenant_id: str,
        user_id: str,
        session_id: str,
        tool_context: Mapping[str, Any] | None = None,
        steering: SteeringInbox | None = None,
    ) -> AgentResponse:
        """Execute the ReactPlanner with memory integration."""
        trace_id = secrets.token_hex(8)
        turn_id = secrets.token_hex(8)


        conscious = {"conscious": []}
        retrieval = {"snippets": []}


        llm_context = {
            "conscious_memories": conscious.get("conscious", []),
            "retrieved_memories": retrieval.get("snippets", []),
        }
        base_tool_context = {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "session_id": session_id,
            "trace_id": trace_id,
            "turn_id": turn_id,
            "task_id": trace_id,
            "is_subagent": False,
            "status_publisher": self._telemetry.publish_status,
        }
        merged_tool_context = {
            **self._tool_context_defaults,
            **(dict(tool_context or {})),
            **base_tool_context,
        }


        result = await self._planner.run(
            query=query,
            llm_context=llm_context,
            tool_context=merged_tool_context,
            steering=steering,
        )

        if isinstance(result, PlannerPause):

            raise EchoAgentFlowError("Planner paused unexpectedly")

        if not isinstance(result, PlannerFinish):
            raise EchoAgentFlowError("Planner did not finish successfully")

        payload: Any = result.payload
        answer_text = _extract_answer(payload)



        return AgentResponse(
            answer=answer_text,
            trace_id=trace_id,
            metadata=dict(result.metadata),

        )



    async def stop(self) -> None:
        """Graceful shutdown hook."""
        if self._started:

            self._started = False
            _LOGGER.info("echo-agent orchestrator stopped")

