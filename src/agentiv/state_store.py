from __future__ import annotations

import json
import os
from typing import Any

import httpx

from penguiflow.state import RemoteBinding, StateStore, StoredEvent
from databricks.sdk import WorkspaceClient
from .util import is_running_locally, get_named_logger, from_env_or_dotenv

logger = get_named_logger(__name__)

def _get_workspace_client() -> WorkspaceClient:
    client_id=os.getenv("DATABRICKS_CLIENT_ID")
    client_secret=os.getenv("DATABRICKS_CLIENT_SECRET")
    return WorkspaceClient(client_id=client_id, client_secret=client_secret)


class PlatformBackendClient:
    def __init__(self, *, base_url: str, timeout_seconds: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        # If running in databricks, lets log in with the client id and secret
        self._ws = _get_workspace_client() if not is_running_locally() else None

    async def post_message(
        self,
        *,
        session_id: str,
        message_type: str,
        content: str,
        metadata: dict[str, Any] | None = None,
        artifacts: Any | None = None,
    ) -> None:
        endpoint = f"{self._base_url}/sessions/{session_id}/messages"
        payload = {
            "session_id": session_id,
            "message_type": message_type,
            "content": content,
            "metadata": metadata or {},
            "artifacts": artifacts,
        }

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            headers = self._ws.config.authenticate() if self._ws else {}

            await client.post(endpoint, json=payload, headers=headers)

    async def fetch_messages(self, *, session_id: str) -> list[dict[str, Any]]:
        endpoint = f"{self._base_url}/sessions/{session_id}/messages"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            headers = self._ws.config.authenticate() if self._ws else {}

            response = await client.get(endpoint, headers=headers)
            response.raise_for_status()
            raw = response.json()

        messages: list[dict[str, Any]] = []

        if isinstance(raw, list):
            messages = [item for item in raw if isinstance(item, dict)]
        elif isinstance(raw, dict):
            entries = raw.get("messages") or raw.get("data") or raw.get("items") or []
            if isinstance(entries, list):
                messages = [item for item in entries if isinstance(item, dict)]

        history: list[dict[str, Any]] = []

        for msg in messages:
            role = str(msg.get("message_type") or msg.get("role") or "").lower()
            if role not in {"user", "assistant"}:
                continue
            content = msg.get("content") or msg.get("text") or msg.get("message")
            if not content:
                continue
            history.append({"role": role, "content": str(content), "metadata": msg.get("metadata"), "artifacts": msg.get("artifacts")})

        return history


class IcebergClient:
    def __init__(self, *, base_url: str, api_key: str, tenant_id: str, timeout_seconds: float = 120.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._tenant_id = tenant_id
        self._timeout = timeout_seconds
        # If running in databricks, lets log in with the client id and secret
        self._ws = _get_workspace_client() if not is_running_locally() else None

    async def ingest_interaction(
        self,
        *,
        session_id: str,
        user_id: str,
        user_prompt: str,
        agent_response: str,
        agent_interaction_id: str | None = None,
    ) -> None:
        endpoint = f"{self._base_url}/memory/ingest_interaction"
        payload = {
            "tenant_id": self._tenant_id,
            "user_id": user_id,
            "session_id": session_id,
            "user_prompt": user_prompt,
            "agent_response": agent_response,
            "source": "chat",
        }
        if agent_interaction_id:
            payload["agent_interaction_id"] = agent_interaction_id
        headers = {
            "X-Pengui-Tenant": self._tenant_id,
            "X-Pengui-User": user_id,
            "Authorization": f"Bearer {self._api_key}" if self._api_key else "",
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            await client.post(endpoint, json=payload, headers=headers)


class AgentivStateStore(StateStore):
    """Adapter that maps save/load memory state to platform backend messages."""

    def __init__(self, platform_client: PlatformBackendClient, iceberg_client: IcebergClient | None) -> None:
        self._platform_client = platform_client
        self._iceberg_client = iceberg_client

    async def save_event(self, event: StoredEvent) -> None:
        trace_id = event.trace_id or "global"
        payload = {
            "trace_id": event.trace_id,
            "ts": event.ts,
            "kind": event.kind,
            "node_name": event.node_name,
            "node_id": event.node_id,
            "payload": dict(event.payload),
        }
        await self.save_memory_state(
            trace_id,
            {
                "session_id": trace_id,
                "message_type": "assistant",
                "content": json.dumps(payload, ensure_ascii=True),
                "metadata": {
                    "state_store": "save_event",
                    "trace_id": event.trace_id,
                    "kind": event.kind,
                    "node_name": event.node_name,
                    "node_id": event.node_id,
                    "ts": event.ts,
                },
                "artifacts": None,
            },
        )

    async def load_history(self, trace_id: str) -> list[StoredEvent]:
        state = await self.load_memory_state(trace_id)
        messages = state.get("messages") if isinstance(state, dict) else None
        if not isinstance(messages, list):
            return []
        events: list[StoredEvent] = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            metadata = message.get("metadata") if isinstance(message.get("metadata"), dict) else {}
            if metadata.get("state_store") != "save_event":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            events.append(
                StoredEvent(
                    trace_id=parsed.get("trace_id"),
                    ts=float(parsed.get("ts", 0.0)),
                    kind=str(parsed.get("kind", "")),
                    node_name=parsed.get("node_name"),
                    node_id=parsed.get("node_id"),
                    payload=parsed.get("payload") or {},
                )
            )
        return events

    async def save_remote_binding(self, binding: RemoteBinding) -> None:
        return None

    async def save_memory_state(self, key: str, state: dict[str, Any]) -> None:
        session_id = state.get("session_id") or key
        if not isinstance(session_id, str) or not session_id:
            return
        await self._platform_client.post_message(
            session_id=session_id,
            message_type=str(state.get("message_type") or "assistant"),
            content=str(state.get("content") or ""),
            metadata=state.get("metadata") if isinstance(state.get("metadata"), dict) else None,
            artifacts=state.get("artifacts"),
        )
        if self._iceberg_client is None:
            return
        if str(state.get("message_type")) != "assistant":
            return
        user_prompt = state.get("user_prompt")
        agent_response = state.get("content")
        metadata = state.get("metadata") if isinstance(state.get("metadata"), dict) else {}
        user_id = metadata.get("user_id")
        if not isinstance(user_prompt, str) or not isinstance(agent_response, str) or not isinstance(user_id, str):
            return
        await self._iceberg_client.ingest_interaction(
            session_id=session_id,
            user_id=user_id,
            user_prompt=user_prompt,
            agent_response=agent_response,
            agent_interaction_id=state.get("agent_interaction_id"),
        )

    async def load_memory_state(self, key: str) -> dict[str, Any]:
        history = await self._platform_client.fetch_messages(session_id=key)
        return {"messages": history}


def build_agentiv_state_store_from_env() -> StateStore:
    platform_url = from_env_or_dotenv("PLATFORM_URL", "")
    
    if not platform_url:
        raise ValueError("You need to specify PLATFORM_URL env variable")

    platform_timeout = float(from_env_or_dotenv("PLATFORM_TIMEOUT_SECONDS", "10"))
    platform_client = PlatformBackendClient(base_url=platform_url, timeout_seconds=platform_timeout)

    iceberg_enabled = from_env_or_dotenv("PENGUI_ICEBERG_ENABLED", "").lower() in {"1", "true", "yes", "on"}
    iceberg_url = from_env_or_dotenv("PENGUI_ICEBERG_URL", "")
    iceberg_api_key = from_env_or_dotenv("PENGUI_ICEBERG_API_KEY", "")
    iceberg_tenant_id = from_env_or_dotenv("PENGUI_ICEBERG_TENANT_ID", "")
    iceberg_timeout = float(from_env_or_dotenv("PENGUI_ICEBERG_TIMEOUT_SECONDS", "20"))

    iceberg_client = None

    if iceberg_enabled:

        if not iceberg_url:
            raise ValueError("You need to specify PENGUI_ICEBERG_URL env variable")
        
        if not iceberg_api_key:
            raise ValueError("You need to specify PENGUI_ICEBERG_API_KEY env variable")
        
        if not iceberg_tenant_id:
            raise ValueError("You need to specify PENGUI_ICEBERG_TENANT_ID env variable")
        
        iceberg_client = IcebergClient(
            base_url=iceberg_url,
            api_key=iceberg_api_key,
            tenant_id=iceberg_tenant_id,
            timeout_seconds=iceberg_timeout,
        )

    return AgentivStateStore(platform_client, iceberg_client)
