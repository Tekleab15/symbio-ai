from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from .envelope_codec import envelope_from_band_content
from .models import CropCase, SymbioEnvelope
from .services.band import BandRestTransport
from .services.phoenix import PhoenixChannelClient, PhoenixEvent
from .settings import settings
from .storage import store

logger = logging.getLogger("symbio.supervisor")


@dataclass
class CaseHeartbeat:
    case_id: str
    last_seen_monotonic: float
    last_envelope: SymbioEnvelope
    escalated_for_event_id: str | None = None


class RoomWatchdogSupervisor:
    def __init__(self) -> None:
        self.chat_id = settings.band_case_room_id
        self.human_api_key = settings.band_human_api_key
        self.transport = BandRestTransport()
        self.heartbeats: dict[str, CaseHeartbeat] = {}
        self._closed = asyncio.Event()

    async def run_forever(self) -> None:
        if not self.chat_id:
            raise RuntimeError("BAND_CASE_ROOM_ID is required for Supervisor watchdog")
        if not self.human_api_key:
            raise RuntimeError("BAND_HUMAN_API_KEY is required for room-wide Supervisor watchdog")
        socket = PhoenixChannelClient(api_key=self.human_api_key)
        await socket.connect()
        monitor_task = asyncio.create_task(self._monitor_loop())
        try:
            await socket.join(f"chat_room:{self.chat_id}")
            async for event in socket.events():
                await self._handle_socket_event(event)
        finally:
            self._closed.set()
            monitor_task.cancel()
            await socket.close()

    async def _handle_socket_event(self, event: PhoenixEvent) -> None:
        if event.event == "message_created":
            envelope = envelope_from_band_content(event.payload.get("content", ""))
            if not envelope:
                return
            self._record_heartbeat(envelope)
            if envelope.task_state == "agent_failed":
                await self._escalate(envelope, reason="agent_failed_message")
        elif event.event == "event_created":
            metadata = event.payload.get("metadata") or {}
            case_id = metadata.get("case_id")
            if event.payload.get("message_type") == "error" and case_id:
                synthetic = self._synthetic_envelope_from_error(event.payload)
                await self._escalate(synthetic, reason="band_error_event")

    def _record_heartbeat(self, envelope: SymbioEnvelope) -> None:
        if envelope.agent == "Supervisor_Agent":
            return
        self.heartbeats[envelope.case_id] = CaseHeartbeat(
            case_id=envelope.case_id,
            last_seen_monotonic=time.monotonic(),
            last_envelope=envelope,
            escalated_for_event_id=self.heartbeats.get(envelope.case_id, CaseHeartbeat(envelope.case_id, 0, envelope)).escalated_for_event_id,
        )

    async def _monitor_loop(self) -> None:
        while not self._closed.is_set():
            await asyncio.sleep(settings.supervisor_scan_seconds)
            now = time.monotonic()
            for heartbeat in list(self.heartbeats.values()):
                envelope = heartbeat.last_envelope
                if envelope.task_state in {"complete", "human_review_required", "approved", "rejected", "supervisor_escalated"}:
                    continue
                if heartbeat.escalated_for_event_id == envelope.event_id:
                    continue
                age = now - heartbeat.last_seen_monotonic
                if age >= settings.supervisor_timeout_seconds:
                    await self._escalate(envelope, reason=f"no_downstream_handoff_for_{int(age)}s")
                    heartbeat.escalated_for_event_id = envelope.event_id

    async def _escalate(self, envelope: SymbioEnvelope, *, reason: str) -> None:
        try:
            case = self._load_case(envelope)
        except Exception:
            case = CropCase(
                case_id=envelope.case_id,
                crop="unknown", location="unknown", symptoms=[],
                metadata={"created_by": "Supervisor_Agent synthetic recovery"},
            )
        failure_payload = {
            "reason": reason, "stalled_after_agent": envelope.agent,
            "last_event_id": envelope.event_id, "last_task_state": envelope.task_state,
            "expected_next_agent": envelope.next_agent, "supervisor_timeout_seconds": settings.supervisor_timeout_seconds,
        }
        escalation = SymbioEnvelope(
            case_id=envelope.case_id,
            agent="Supervisor_Agent",
            role="room_watchdog_supervisor",
            task_state="dead_drop_detected",
            finding=f"Supervisor detected a stalled Band workflow after {envelope.agent}: {reason}.",
            next_agent="Human_Reviewer",
            handoff_reason="Room-wide watchdog observed no valid downstream handoff before timeout.",
            input_refs=[envelope.event_id], confidence=0.88, risk_level="high", requires_human_review=True,
            payload=failure_payload,
            context_snapshot={"case": case.to_dict(), "failure": failure_payload},
        )
        delivery = await self.transport.send_envelope(escalation)
        escalation.band_delivery = delivery.to_dict()
        store.append_event(escalation)
        case.status = "supervisor_escalated"
        case.requires_human_review = True
        case.risk_level = "high"
        store.save_case(case)
        logger.warning("Supervisor escalated stalled case %s after %s", envelope.case_id, envelope.agent)

    @staticmethod
    def _load_case(envelope: SymbioEnvelope) -> CropCase:
        case_data = (envelope.context_snapshot or {}).get("case")
        if isinstance(case_data, dict):
            return CropCase.from_dict(case_data)
        return store.get_case(envelope.case_id)

    @staticmethod
    def _synthetic_envelope_from_error(payload: dict[str, Any]) -> SymbioEnvelope:
        metadata = payload.get("metadata") or {}
        case_id = metadata.get("case_id") or "unknown-case"
        failed_agent = metadata.get("failed_agent") or payload.get("sender_name") or "unknown_agent"
        return SymbioEnvelope(
            case_id=case_id,
            agent=failed_agent,
            role="band_error_event",
            task_state="agent_failed",
            finding=payload.get("content", "Band error event observed."),
            next_agent="Supervisor_Agent",
            input_refs=[metadata.get("failed_event_id") or payload.get("id") or "unknown"],
            confidence=0.0, risk_level="high", requires_human_review=True,
            payload={"error": payload.get("content", "error event"), **metadata},
            context_snapshot={"failure": {"error": payload.get("content", "error event"), **metadata}},
        )