from __future__ import annotations

import asyncio
import contextlib
import json
import os
import secrets
from collections.abc import AsyncIterator, Callable, Coroutine, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator, Protocol

from ag_ui.core import RunAgentInput
from fastapi import FastAPI, WebSocket
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect


from penguiflow.state import StateStore
from penguiflow.agui_adapter.base import AGUIAdapter, AGUIEvent, generate_id
from penguiflow.planner import PlannerEvent
from .util import get_named_logger

logger = get_named_logger(__name__)

class StatusUpdateProtocol(Protocol):
    """Protocol for status update payloads."""

    @property
    def status(self) -> str: ...

    @property
    def message(self) -> str: ...

    @property
    def timestamp(self) -> datetime: ...


class AgentResponseProtocol(Protocol):
    """Protocol for orchestrator response envelopes."""

    @property
    def answer(self) -> str | None: ...

    @property
    def trace_id(self) -> str: ...

    @property
    def metadata(self) -> dict[str, Any] | None: ...


class AgentTelemetryProtocol(Protocol):
    """Protocol for telemetry implementations."""

    def record_planner_event(self, event: PlannerEvent) -> None: ...

    def publish_status(self, update: StatusUpdateProtocol) -> None: ...


class OrchestratorProtocol(Protocol):
    """Protocol for orchestrator implementations."""

    async def execute(
        self,
        query: str,
        *,
        tenant_id: str,
        user_id: str,
        session_id: str,
        #conversation_history: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AgentResponseProtocol: ...

    async def stop(self) -> None: ...


OrchestratorFactory = Callable[[AgentTelemetryProtocol], OrchestratorProtocol]


@dataclass
class _ConnectionManager:
    active_connections: dict[str, WebSocket] = field(default_factory=dict)

    async def connect(self, session_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str) -> None:
        self.active_connections.pop(session_id, None)

    async def send_json(self, session_id: str, payload: Any) -> None:
        websocket = self.active_connections.get(session_id)
        if websocket is None:
            return
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump()
        await websocket.send_json(payload)


class _TelemetryBridge:
    """Standalone telemetry bridge implementing AgentTelemetryProtocol."""

    def __init__(self, *, flow_name: str) -> None:
        self.flow_name = flow_name
        self._event_callback: Callable[[PlannerEvent], None] | None = None
        self._status_callback: Callable[[str | None, int | None], None] | None = None

    @contextlib.contextmanager
    def subscribe(
        self,
        *,
        status_callback: Callable[[str | None, int | None], None],
        event_callback: Callable[[PlannerEvent], None],
    ) -> Iterator[None]:
        # Save previous callbacks (if any)
        prev_event = self._event_callback
        prev_status = self._status_callback

        # Install the new callbacks
        self._event_callback = event_callback
        self._status_callback = status_callback
        try:
            yield None # Agent runs while we're "inside" the with block
        finally:
            # Restore previous callbacks
            self._event_callback = prev_event
            self._status_callback = prev_status

    def record_planner_event(self, event: PlannerEvent) -> None:
        # When the ReactPlanner emits a PlannerEvent, it calls this function.
        logger.debug("planner_event flow=%s type=%s", self.flow_name, event.event_type)
        if self._event_callback:
            self._event_callback(event)

    def publish_status(self, update: StatusUpdateProtocol) -> None:
        logger.debug("status_update flow=%s msg=%s", self.flow_name, update.message)
        if self._status_callback:
            self._status_callback(update.message, None)


@dataclass
class _Session:
    orchestrator: OrchestratorProtocol
    telemetry: _TelemetryBridge
    history: list[dict[str, Any]] = field(default_factory=list)
    last_user_message: str | None = None
    last_user_id: str | None = None
    last_run_id: str | None = None


class _SessionRegistry:
    def __init__(self, *, orchestrator_factory: OrchestratorFactory, flow_name: str) -> None:
        self._orchestrator_factory = orchestrator_factory
        self._sessions: dict[str, _Session] = {}
        self._flow_name=flow_name

    def start_session(self, session_id: str) -> _Session:
        telemetry = _TelemetryBridge(flow_name=self._flow_name)
        orchestrator = self._orchestrator_factory(telemetry)
        session = _Session(orchestrator=orchestrator, telemetry=telemetry)
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> _Session | None:
        return self._sessions.get(session_id)

    async def close(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        await session.orchestrator.stop()


class _SessionService:
    def __init__(
        self,
        registry: _SessionRegistry,
        state_store: StateStore | None,
    ) -> None:
        self._registry = registry
        self._state_store = state_store

    async def start(self, session_id: str) -> _Session:
        session = self._registry.start_session(session_id)
        if self._state_store is not None:
            state = await self._state_store.load_memory_state(session_id)
            history = state.get("messages") if isinstance(state, dict) else None
            if isinstance(history, list):
                session.history = history
        return session

    async def send_message(
        self,
        *,
        session_id: str,
        message: str,
        tenant_id: str,
        user_id: str,
    ) -> AgentResponseProtocol:

        session = self._registry.get(session_id)
        if session is None:
            session = self._registry.start_session(session_id)
        self._save_message_state(
            session_id=session_id,
            message_type="user",
            content=message,
            metadata={"tenant_id": tenant_id, "user_id": user_id},
        )
        session.last_user_message = message
        session.last_user_id = user_id
        return await session.orchestrator.execute(
            query=message,
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            #conversation_history=session.history,
        )

    async def close(self, session_id: str) -> None:
        await self._registry.close(session_id)

    def _save_message_state(
        self,
        *,
        session_id: str,
        message_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        artifacts: Any | None = None,
        extra_state: dict[str, Any] | None = None,
    ) -> None:
        if self._state_store is None:
            return

        async def _post() -> None:
            payload = {
                "session_id": session_id,
                "message_type": message_type,
                "content": content,
                "metadata": metadata or {},
                "artifacts": artifacts,
            }
            if extra_state:
                payload.update(extra_state)
            await self._state_store.save_memory_state(session_id, payload)

        _create_background_task(_post(), session_id=session_id)


class _AGUIWebsocketAdapter(AGUIAdapter):
    """Minimal adapter to map planner events to AG-UI events."""

    def __init__(self) -> None:
        super().__init__()
        self.streamed_answer = False

    async def run(self, _input: RunAgentInput) -> AsyncIterator[AGUIEvent]:  # type: ignore[override]
        raise NotImplementedError("Websocket adapter does not execute runs directly.")

    def start_run(self, _input: RunAgentInput) -> None:
        self.streamed_answer = False

    def end_run(self) -> None:
        self.streamed_answer = False

    def convert_planner_event(self, event: PlannerEvent) -> list[AGUIEvent]:
        extra = dict(event.extra or {})
        mapped: list[AGUIEvent] = []

        if event.event_type == "step_start":
            step_name = extra.get("step_name") or event.node_name or f"step_{event.trajectory_step}"
            mapped.append(self.step_start(step_name, **extra))
            return mapped

        if event.event_type == "step_complete":
            step_name = event.node_name or extra.get("step_name") or f"step_{event.trajectory_step}"
            mapped.append(self.step_end(step_name, **extra))
            return mapped

        if event.event_type == "tool_start":
            tool_name = extra.get("tool_name") or event.node_name or "tool"
            tool_call_id = extra.get("tool_call_id") or generate_id("call")
            mapped.append(self.tool_start(tool_name, tool_call_id=tool_call_id))
            if extra.get("args") is not None:
                mapped.append(self.tool_args(tool_call_id, json.dumps(extra.get("args"), ensure_ascii=True, default=str)))
                mapped.append(self.tool_end(tool_call_id))
            return mapped

        if event.event_type == "tool_result":
            tool_call_id = extra.get("tool_call_id") or generate_id("call")
            content = extra.get("result") or extra.get("output") or ""
            mapped.append(self.tool_result(tool_call_id, content))
            return mapped

        if event.event_type == "llm_stream_chunk":
            channel = extra.get("channel")
            text = str(extra.get("text") or "")
            done = bool(extra.get("done"))
            phase = extra.get("phase")
            if channel == "answer" and text:
                if not self._message_started:
                    mapped.append(self.text_start())
                mapped.append(self.text_content(text))
                if done:
                    mapped.append(self.text_end())
                self.streamed_answer = True
                return mapped
            if channel == "thinking" and text:
                mapped.append(self.custom("thinking", {"text": text, "phase": phase, "done": done}))
                return mapped
            if channel == "revision" and text:
                mapped.append(self.custom("revision", {"text": text, "done": done}))
                return mapped
            return mapped

        if event.event_type in {"artifact_chunk", "artifact_stored"}:
            mapped.append(self.custom(event.event_type, extra))
            return mapped

        return mapped

    def emit_text_block(self, text: str) -> Iterable[AGUIEvent]:
        if not text:
            return []
        events: list[AGUIEvent] = []
        if not self._message_started:
            events.append(self.text_start())
        events.append(self.text_content(text))
        events.append(self.text_end())
        self.streamed_answer = True
        return events




class _AguiWebsocketOutputStrategy:
    """Emit AG-UI events over the existing WebSocket connection."""

    def __init__(
        self,
        *,
        sender: Callable[[str, Any], Any],
        session_service: _SessionService,
        session_registry: _SessionRegistry,
    ) -> None:
        self._send = sender
        self._session_service = session_service
        self._session_registry = session_registry

    async def on_connect(self, session_id: str) -> None:
        logger.info("AG-UI websocket connected: %s", session_id)

    async def on_message(self, session_id: str, message: Any) -> None:
        # Ignore run cancel from front
        if isinstance(message, dict) and message.get("type") == "run.cancel":
            logger.info("Ignoring run.cancel for session %s", session_id)
            return
        run_input = self._parse_input(session_id, message)
        if run_input is None:
            await self._send_error(session_id, "Invalid AG-UI input")
            return

        # Extract the "user" message
        query = _pick_query(run_input.messages)
        if not query:
            await self._send_error(session_id, "Missing user message")
            return

        # Upsert session
        session = self._session_registry.get(session_id)
        if session is None:
            session = self._session_registry.start_session(session_id)

        
        telemetry = session.telemetry
        session.last_run_id = run_input.run_id

        adapter = _AGUIWebsocketAdapter()
        adapter.start_run(run_input)

        queue: asyncio.Queue[Any] = asyncio.Queue()
        sentinel = object()
        run_error: Exception | None = None
        run_result: AgentResponseProtocol | None = None
        artifact_emitted = False

        def event_callback(event: PlannerEvent) -> None:
            nonlocal artifact_emitted
            event_type = getattr(event, "event_type", None)
            if isinstance(event_type, str) and event_type in {"artifact_chunk", "artifact_stored"}:
                artifact_emitted = True
                self._debug("AG-UI planner artifact event: %s", event_type)
            for mapped in adapter.convert_planner_event(event):
                queue.put_nowait(mapped)

        def status_callback(msg: str | None, step: int | None) -> None:
            payload = {"message": msg, "step": step}
            queue.put_nowait(adapter.custom("status", payload))

        async def run_agent() -> None:
            nonlocal run_error, run_result
            try:
                forwarded_props = run_input.forwarded_props or {}
                metadata = forwarded_props.get("metadata") if isinstance(forwarded_props, dict) else None
                tenant_id = _safe_str(metadata.get("tenant_id") if isinstance(metadata, dict) else None, "default-tenant")
                user_id = _safe_str(metadata.get("user_id") if isinstance(metadata, dict) else None, "default-user")
                run_result = await self._session_service.send_message(
                    session_id=session_id,
                    message=query,
                    tenant_id=tenant_id,
                    user_id=user_id,
                )
                if run_result is not None and run_result.answer:
                    session = self._session_service._registry.get(session_id)
                    if session is not None and session.last_user_message:
                        artifacts = None
                        if run_result.metadata and isinstance(run_result.metadata, dict):
                            artifacts = run_result.metadata.get("artifacts")
                        self._session_service._save_message_state(
                            session_id=session_id,
                            message_type="assistant",
                            content=run_result.answer,
                            metadata={"tenant_id": tenant_id, "user_id": user_id},
                            artifacts=artifacts,
                            extra_state={
                                "user_prompt": session.last_user_message,
                                "agent_interaction_id": session.last_run_id,
                            },
                        )
            except Exception as exc:
                run_error = exc
            finally:
                queue.put_nowait(sentinel)

        asyncio.create_task(run_agent())

        async def stream_events():
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                yield item

            if run_error is not None:
                raise run_error

            if run_result is not None and not artifact_emitted:
                metadata = run_result.metadata or {}
                artifacts = metadata.get("artifacts") if isinstance(metadata, dict) else None
                payloads = _normalize_artifacts_for_agui(artifacts)
                self._debug("AG-UI artifact fallback metadata=%s", _safe_json(metadata))
                self._debug("AG-UI artifact fallback normalized=%s", _safe_json(payloads))
                if not payloads:
                    self._info("AG-UI artifact fallback empty; metadata=%s", _safe_json(metadata))
                for payload in payloads:
                    self._info("AG-UI artifact fallback sending=%s", _safe_json(payload))
                    yield adapter.custom("artifact_chunk", payload)

            if run_result is not None and run_result.answer and not adapter.streamed_answer:
                for event in adapter.emit_text_block(run_result.answer):
                    yield event

        with telemetry.subscribe(status_callback=status_callback, event_callback=event_callback):
            try:
                async for event in adapter.with_run_lifecycle(run_input, stream_events()):
                    await self._send_event(session_id, event)
            except Exception as exc:
                logger.warning("AG-UI run failed: %s", exc)
            finally:
                adapter.end_run()

    async def on_disconnect(self, session_id: str) -> None:
        logger.info("AG-UI websocket disconnected: %s", session_id)

    def _parse_input(self, session_id: str, message: Any) -> RunAgentInput | None:
        if isinstance(message, RunAgentInput):
            return message
        try:
            return RunAgentInput.model_validate(message)
        except ValidationError:
            pass
        if isinstance(message, dict):
            normalized = _normalize_message_payload(message)
            if normalized is not None:
                try:
                    return RunAgentInput.model_validate(normalized)
                except ValidationError:
                    pass
        if isinstance(message, dict):
            user_message = message.get("message")
            if not isinstance(user_message, str):
                user_message = message.get("text")
            if not isinstance(user_message, str):
                return None
            forwarded_props = None
            metadata = message.get("metadata")
            if isinstance(metadata, dict):
                forwarded_props = {"metadata": metadata}
            return _build_run_input(
                session_id=session_id,
                message=user_message,
                run_id=secrets.token_hex(8),
                forwarded_props=forwarded_props,
            )
        return None

    async def _send_event(self, session_id: str, event: Any) -> None:
        if hasattr(event, "model_dump"):
            payload = event.model_dump(by_alias=True, exclude_none=True)
        else:
            payload = event
        if isinstance(payload, dict):
            if payload.get("type") == "CUSTOM" and payload.get("name") in {"artifact_chunk", "artifact_stored"}:
                self._info("AG-UI sending artifact event=%s", _safe_json(payload))
        await self._send(session_id, payload)

    async def _send_error(self, session_id: str, message: str) -> None:
        payload = {"type": "RUN_ERROR", "message": message}
        await self._send(session_id, payload)

    def _info(self, message: str, *args: Any) -> None:
        logger.info(message, *args)

    def _debug(self, message: str, *args: Any) -> None:
        logger.debug(message, *args)


class _WebsocketChatService:
    def __init__(
        self,
        *,
        connection_manager: _ConnectionManager,
        session_service: _SessionService,
        output_strategy: _AguiWebsocketOutputStrategy,
        telemetry_factory: Callable[[], AgentTelemetryProtocol],
    ) -> None:
        self._manager = connection_manager
        self._session_service = session_service
        self._output_strategy = output_strategy
        self._telemetry_factory = telemetry_factory

    async def handle_session(self, websocket: WebSocket, session_id: str) -> None:
        await self._manager.connect(session_id, websocket)
        logger.info("WebSocket connected; starting session for %s", session_id)
        await self._session_service.start(session_id)
        await self._output_strategy.on_connect(session_id)

        keepalive_task = asyncio.create_task(self._keepalive_loop(session_id))
        try:
            await self._receive_loop(session_id)
        except WebSocketDisconnect:
            self._manager.disconnect(session_id)
        finally:
            keepalive_task.cancel()
            await self._output_strategy.on_disconnect(session_id)
            await self._session_service.close(session_id)

    async def _receive_loop(self, session_id: str) -> None:
        websocket = self._manager.active_connections[session_id]
        recv_task: asyncio.Task[Any] | None = None

        while True:
            try:
                if recv_task is None:
                    recv_task = asyncio.create_task(websocket.receive_json())
                done, _ = await asyncio.wait({recv_task}, timeout=_receive_timeout_seconds())
                if not done:
                    await self._send(session_id, {"type": "ping"})
                    continue
                message = recv_task.result()
                recv_task = None
            except asyncio.TimeoutError:
                await self._send(session_id, {"type": "ping"})
                continue

            if isinstance(message, dict) and message.get("type") == "pong":
                continue
            if isinstance(message, dict) and message.get("type") == "ping":
                await self._send(session_id, {"type": "pong"})
                continue

            await self._handle_message(session_id, message)

    async def _handle_message(self, session_id: str, message: Any) -> None:
        self._log_payload("PAYLOAD_REV_FROM_FRONT", session_id, message)
        await self._output_strategy.on_message(session_id, message)

    async def _keepalive_loop(self, session_id: str) -> None:
        while True:
            await asyncio.sleep(_keepalive_interval_seconds())
            await self._send(session_id, {"type": "ping"})

    async def _send(self, session_id: str, payload: Any) -> None:
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump()
        self._log_payload("PAYLOAD_REV_TO_FRONT", session_id, payload)
        await self._manager.send_json(session_id, payload)

    def _log_payload(self, label: str, session_id: str, payload: Any) -> None:
        if _is_thinking_payload(payload):
            return
        logger.info(
            "%s session_id=%s payload=%s",
            label,
            session_id,
            _serialize_payload(payload),
        )


def _receive_timeout_seconds() -> float:
    return float(os.getenv("WEBSOCKET_RECEIVE_TIMEOUT", "30"))


def _keepalive_interval_seconds() -> float:
    return float(os.getenv("WEBSOCKET_KEEPALIVE_INTERVAL", "10"))


def _safe_str(value: Any, fallback: str) -> str:
    return value if isinstance(value, str) and value.strip() else fallback


def _pick_query(messages: Iterable[Any]) -> str:
    for msg in reversed(list(messages or [])):
        role = getattr(msg, "role", None)
        content = getattr(msg, "content", None)
        if role is None and isinstance(msg, Mapping):
            role = msg.get("role")
            content = msg.get("content")
        if role != "user":
            continue
        text = _extract_text_content(content)
        if text:
            return text
    return ""


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Mapping):
        text = content.get("text")
        if isinstance(text, str):
            return text
    text_attr = getattr(content, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    if isinstance(content, Iterable) and not isinstance(content, (str, bytes, Mapping)):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                    continue
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
        if parts:
            return "\n".join(parts)
    return ""


def _serialize_payload(payload: Any) -> str:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump()
    try:
        return json.dumps(payload, ensure_ascii=True, default=str)
    except TypeError:
        return str(payload)


def _is_thinking_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("status") == "thinking":
        return True
    if payload.get("type") == "CUSTOM" and payload.get("name") == "thinking":
        return True
    if payload.get("isThinking") is True:
        return True
    return False


def _normalize_artifacts_for_agui(artifacts: Any) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for content in _iter_artifact_contents(artifacts):
        payload = _normalize_single_artifact_for_agui(content)
        if payload is not None:
            payloads.append(payload)
    return payloads


def _normalize_single_artifact_for_agui(artifacts: dict[str, Any]) -> dict[str, Any] | None:
    component = artifacts.get("component")
    if isinstance(component, str) and component:
        props = artifacts.get("props") if isinstance(artifacts.get("props"), dict) else {}
        chunk = {
            "component": component,
            "props": props,
            "title": artifacts.get("title"),
            "description": artifacts.get("description"),
        }
        return {
            "artifact_type": "ui_component",
            "chunk": chunk,
            "done": True,
        }

    artifact_type = artifacts.get("type")
    if isinstance(artifact_type, str) and artifact_type:
        props = artifacts.get("props") if isinstance(artifacts.get("props"), dict) else {}
        chunk = {
            "component": artifact_type,
            "props": props,
            "title": artifacts.get("title"),
            "description": artifacts.get("description"),
        }
        return {
            "artifact_type": "ui_component",
            "chunk": chunk,
            "done": True,
        }

    if artifacts:
        return {"artifact_type": "raw", "chunk": artifacts, "done": True}

    return None


def _iter_artifact_contents(artifacts: Any) -> list[dict[str, Any]]:
    if isinstance(artifacts, list):
        entries = [item for item in artifacts if isinstance(item, dict)]
    elif isinstance(artifacts, dict):
        entries = [artifacts]
    else:
        return []

    ordered = _order_artifact_entries(entries)
    contents: list[dict[str, Any]] = []
    for entry in ordered:
        content = entry.get("artifact_content")
        if isinstance(content, dict):
            contents.append(content)
            continue
        if "type" in entry or "component" in entry:
            contents.append(entry)
    return contents


def _order_artifact_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    has_sort = any(isinstance(entry.get("sort_order"), (int, float)) for entry in entries)
    if not has_sort:
        return entries
    indexed = list(enumerate(entries))

    def key(item: tuple[int, dict[str, Any]]) -> tuple[int, float, int]:
        idx, entry = item
        order = entry.get("sort_order")
        if isinstance(order, (int, float)):
            return (0, float(order), idx)
        return (1, 0.0, idx)

    return [entry for _, entry in sorted(indexed, key=key)]


def _safe_json(value: Any, *, max_chars: int = 2000) -> str:
    try:
        encoded = json.dumps(value, ensure_ascii=True, default=str)
    except TypeError:
        encoded = str(value)
    if len(encoded) > max_chars:
        return f"{encoded[: max_chars - 12]}...<truncated>"
    return encoded


def _create_background_task(coro: asyncio.Future[Any] | Coroutine[Any, Any, Any], *, session_id: str) -> None:
    try:
        task = asyncio.create_task(coro)
    except RuntimeError:
        logger.warning(
            "No running event loop to send platform messages for %s",
            session_id,
        )
        return

    def _log_failure(future: asyncio.Future[Any]) -> None:
        try:
            future.result()
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.warning("Platform sync failed for %s: %s", session_id, exc)

    task.add_done_callback(_log_failure)

def _normalize_message_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return None
    normalized_messages: list[Any] = []
    changed = False
    for item in messages:
        if not isinstance(item, dict):
            normalized_messages.append(item)
            continue
        normalized = dict(item)
        role = normalized.get("role")
        if role == "agent":
            normalized["role"] = "assistant"
            changed = True
        if "content" not in normalized:
            text = normalized.get("text")
            if isinstance(text, str):
                normalized["content"] = text
                changed = True
        normalized_messages.append(normalized)
    if not changed:
        return payload
    updated = dict(payload)
    updated["messages"] = normalized_messages
    return updated


def _build_run_input(
    *,
    session_id: str,
    message: str,
    run_id: str,
    forwarded_props: Any | None,
) -> RunAgentInput:
    payload = {
        "threadId": session_id,
        "runId": run_id,
        "state": {},
        "messages": [
            {
                "id": generate_id("msg"),
                "role": "user",
                "content": message,
            }
        ],
        "tools": [],
        "context": [],
        "forwardedProps": forwarded_props or {},
    }
    return RunAgentInput.model_validate(payload)


def mount_agui_websocket(
    app: FastAPI,
    *,
    state_store: StateStore,
    flow_name: str,
    orchestrator_factory: OrchestratorFactory,
) -> None:
    manager = _ConnectionManager()
    session_registry = _SessionRegistry(orchestrator_factory=orchestrator_factory, flow_name=flow_name)
    session_service = _SessionService(session_registry, state_store)

    output_strategy = _AguiWebsocketOutputStrategy(
        sender=manager.send_json,
        session_service=session_service,
        session_registry=session_registry
    )

    chat_service = _WebsocketChatService(
        connection_manager=manager,
        session_service=session_service,
        output_strategy=output_strategy,
        telemetry_factory=lambda: _TelemetryBridge(flow_name=flow_name)
    )

    app.state.chat_service = chat_service

    @app.websocket("/ws/chat/{session_id}")
    async def chat_ws(websocket: WebSocket, session_id: str) -> None:
        await chat_service.handle_session(websocket, session_id)
