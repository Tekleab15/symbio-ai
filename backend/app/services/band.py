from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import httpx

from ..models import SymbioEnvelope, utc_now
from ..settings import settings
from ..storage import store

BandEventType = Literal["tool_call", "tool_result", "thought", "error", "task"]

@dataclass
class BandDelivery:
    mode: str
    status: str
    chat_id: str
    sender: str
    mentions: list[str]
    detail: str = ""
    response: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "status": self.status,
            "chat_id": self.chat_id,
            "sender": self.sender,
            "mentions": self.mentions,
            "detail": self.detail,
            "response": self.response or {},
            "created_at": utc_now(),
        }

class BandTransport:
    async def send_envelope(self, envelope: SymbioEnvelope) -> BandDelivery:
        raise NotImplementedError

    async def record_event(
        self,
        *,
        case_id: str,
        agent: str,
        message_type: BandEventType,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError

class MockBandTransport(BandTransport):
    async def send_envelope(self, envelope: SymbioEnvelope) -> BandDelivery:
        chat_id = f"mock-room-{envelope.case_id}"
        mentions = envelope.mention_targets()
        record = {
            "chat_id": chat_id,
            "sender": envelope.agent,
            "mentions": mentions,
            "content": envelope.to_band_text(),
            "message_type": "text",
            "created_at": utc_now(),
        }
        store.append_band_record(envelope.case_id, record)
        return BandDelivery(
            mode="mock",
            status="recorded",
            chat_id=chat_id,
            sender=envelope.agent,
            mentions=mentions,
            detail="Stored local Band-style transcript. In real mode, agents receive these messages through Band WebSocket @mention routing.",
        )

    async def record_event(
        self,
        *,
        case_id: str,
        agent: str,
        message_type: BandEventType,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = {
            "chat_id": f"mock-room-{case_id}",
            "sender": agent,
            "mentions": [],
            "content": content,
            "message_type": message_type,
            "metadata": metadata or {},
            "created_at": utc_now(),
        }
        store.append_band_record(case_id, record)
        return {"mode": "mock", "status": "recorded", "data": record}

class BandRequestClient:
    def __init__(self, api_key: str, *, base_url: str | None = None) -> None:
        self.api_key = api_key
        self.base_url = (base_url or settings.band_agent_api_base).rstrip("/")

    @property
    def headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key, "Content-Type": "application/json"}

    async def get_me(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(f"{self.base_url}/me", headers=self.headers)
            response.raise_for_status()
            return response.json()

    async def list_chats(self) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(f"{self.base_url}/chats", headers=self.headers)
            response.raise_for_status()
            data = response.json()
            return data.get("data", data if isinstance(data, list) else [])

    async def get_next_message(self, chat_id: str) -> dict[str, Any] | None:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(f"{self.base_url}/chats/{chat_id}/messages/next", headers=self.headers)
            if response.status_code == 204:
                return None
            response.raise_for_status()
            return response.json()

    async def mark_processing(self, chat_id: str, message_id: str) -> None:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(f"{self.base_url}/chats/{chat_id}/messages/{message_id}/processing", headers=self.headers)
            response.raise_for_status()

    async def mark_processed(self, chat_id: str, message_id: str) -> None:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(f"{self.base_url}/chats/{chat_id}/messages/{message_id}/processed", headers=self.headers)
            response.raise_for_status()

    async def mark_failed(self, chat_id: str, message_id: str, detail: str) -> None:
        payload = {"error": {"message": detail[:1000]}}
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(f"{self.base_url}/chats/{chat_id}/messages/{message_id}/failed", headers=self.headers, json=payload)
            response.raise_for_status()

    async def report_activity(self, chat_id: str, working: bool) -> None:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(f"{self.base_url}/chats/{chat_id}/activity", headers=self.headers, json={"working": working})
            if response.status_code not in {200, 201, 204, 404}:
                response.raise_for_status()

    async def send_text_message(self, chat_id: str, content: str, mentions: list[dict[str, str]]) -> dict[str, Any]:
        payload = {"message": {"content": content, "mentions": mentions}}
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(f"{self.base_url}/chats/{chat_id}/messages", headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json() if response.content else {}

    async def create_event(
        self,
        chat_id: str,
        *,
        message_type: BandEventType,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"event": {"content": content, "message_type": message_type}}
        if metadata is not None:
            payload["event"]["metadata"] = metadata
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(f"{self.base_url}/chats/{chat_id}/events", headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json() if response.content else {}

class BandRestTransport(BandTransport):
    def __init__(self) -> None:
        self.chat_id = settings.band_case_room_id
        self.keys = settings.band_agent_keys
        self.participants = settings.band_participants

    async def send_envelope(self, envelope: SymbioEnvelope) -> BandDelivery:
        mentions = envelope.mention_targets()
        if not settings.band_enabled or not self.chat_id:
            return await MockBandTransport().send_envelope(envelope)

        api_key = self.keys.get(envelope.agent)
        if not api_key:
            delivery = BandDelivery(
                mode="band_rest",
                status="failed",
                chat_id=self.chat_id,
                sender=envelope.agent,
                mentions=mentions,
                detail=f"Missing Band API key for sender {envelope.agent}. Add it to BAND_AGENT_KEYS_JSON.",
            )
            store.append_band_record(envelope.case_id, {**delivery.to_dict(), "content": envelope.to_band_text(), "message_type": "text"})
            return delivery

        mention_payload: list[dict[str, str]] = []
        missing: list[str] = []
        for name in mentions:
            participant = self.participants.get(name)
            if participant:
                mention_payload.append(participant)
            else:
                missing.append(name)

        if missing:
            delivery = BandDelivery(
                mode="band_rest",
                status="failed",
                chat_id=self.chat_id,
                sender=envelope.agent,
                mentions=mentions,
                detail=f"Missing participant metadata for mention target(s): {', '.join(missing)}. Add them to BAND_PARTICIPANTS_JSON.",
            )
            store.append_band_record(envelope.case_id, {**delivery.to_dict(), "content": envelope.to_band_text(), "message_type": "text"})
            return delivery

        try:
            response = await BandRequestClient(api_key).send_text_message(self.chat_id, envelope.to_band_text(), mention_payload)
            delivery = BandDelivery(
                mode="band_rest",
                status="sent",
                chat_id=self.chat_id,
                sender=envelope.agent,
                mentions=mentions,
                detail="Delivered to Band Agent API; target agents will be woken by WebSocket @mention routing.",
                response=response,
            )
        except Exception as exc: 
            delivery = BandDelivery(
                mode="band_rest",
                status="failed",
                chat_id=self.chat_id,
                sender=envelope.agent,
                mentions=mentions,
                detail=str(exc),
            )
        store.append_band_record(envelope.case_id, {**delivery.to_dict(), "content": envelope.to_band_text(), "message_type": "text"})
        return delivery

    async def record_event(
        self,
        *,
        case_id: str,
        agent: str,
        message_type: BandEventType,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not settings.band_emit_events:
            return {"status": "disabled"}
        if not settings.band_enabled or not self.chat_id:
            return await MockBandTransport().record_event(case_id=case_id, agent=agent, message_type=message_type, content=content, metadata=metadata)
        api_key = self.keys.get(agent)
        if not api_key:
            record = {
                "mode": "band_event",
                "status": "failed",
                "chat_id": self.chat_id,
                "sender": agent,
                "message_type": message_type,
                "content": content,
                "metadata": metadata or {},
                "detail": f"Missing Band API key for sender {agent}; event was stored locally only.",
                "created_at": utc_now(),
            }
            store.append_band_record(case_id, record)
            return record
        try:
            response = await BandRequestClient(api_key).create_event(self.chat_id, message_type=message_type, content=content, metadata=metadata or {})
            record = {
                "mode": "band_event",
                "status": "sent",
                "chat_id": self.chat_id,
                "sender": agent,
                "message_type": message_type,
                "content": content,
                "metadata": metadata or {},
                "response": response,
                "created_at": utc_now(),
            }
        except Exception as exc:
            record = {
                "mode": "band_event",
                "status": "failed",
                "chat_id": self.chat_id,
                "sender": agent,
                "message_type": message_type,
                "content": content,
                "metadata": metadata or {},
                "detail": str(exc),
                "created_at": utc_now(),
            }
        store.append_band_record(case_id, record)
        return record

def get_band_transport() -> BandTransport:
    if settings.mock_mode or not settings.band_enabled:
        return MockBandTransport()
    return BandRestTransport()