from __future__ import annotations

import asyncio
import logging
from typing import Any

from .agents.agronomy_knowledge import AgronomyKnowledgeAgent
from .agents.base import AgentBase
from .agents.field_intake import FieldIntakeAgent
from .agents.neuro_symbolic_auditor import NeuroSymbolicAuditorAgent
from .agents.operations_report import OperationsReportAgent
from .agents.rule_compliance import RuleComplianceAgent
from .agents.supervisor import SupervisorAgent
from .agents.vision_analysis import VisionAnalysisAgent
from .envelope_codec import (
    band_message_fields,
    case_from_shared_or_store,
    compact_shared_state,
    envelope_from_band_content,
    merge_context_from_events,
)
from .models import CropCase, SymbioEnvelope
from .services.band import BandEventType, BandRequestClient, BandRestTransport, MockBandTransport, get_band_transport
from .services.phoenix import PhoenixChannelClient, PhoenixEvent
from .settings import settings
from .storage import store

logger = logging.getLogger("symbio.agent_runtime")

AGENT_FACTORIES: dict[str, type[AgentBase]] = {
    "Field_Intake_Agent": FieldIntakeAgent,
    "Vision_Analysis_Agent": VisionAnalysisAgent,
    "Agronomy_Knowledge_Agent": AgronomyKnowledgeAgent,
    "Rule_Compliance_Agent": RuleComplianceAgent,
    "Neuro_Symbolic_Auditor": NeuroSymbolicAuditorAgent,
    "Operations_Report_Agent": OperationsReportAgent,
    "Supervisor_Agent": SupervisorAgent,
}

CONDUCTOR_AGENT = "Symbio_Conductor"
FINAL_HUMAN_REVIEW_TARGET = "Human_Reviewer"
TERMINAL_STATES = {"complete", "human_review_required", "approved", "rejected", "supervisor_escalated"}


class AgentWorker:
    def __init__(self, agent_name: str) -> None:
        if agent_name not in AGENT_FACTORIES:
            raise ValueError(f"Unknown Symbio agent: {agent_name}")
        self.agent_name = agent_name
        self.agent: AgentBase = AGENT_FACTORIES[agent_name]()
        self.api_key = settings.band_agent_keys.get(agent_name, "")
        self.agent_id = settings.band_agent_ids.get(agent_name, "")
        self.request_client = BandRequestClient(self.api_key) if self.api_key else None
        self.transport = BandRestTransport() if settings.band_enabled else MockBandTransport()
        self.joined_chats: set[str] = set()

    async def handle_envelope(
        self,
        incoming: SymbioEnvelope,
        *,
        chat_id: str | None = None,
        message_id: str | None = None,
        raw_message: dict[str, Any] | None = None,
    ) -> SymbioEnvelope | None:
        if self.agent_name not in incoming.mention_targets() and self.agent_name != incoming.next_agent:
            logger.debug("%s ignored message %s because it was not mentioned", self.agent_name, incoming.event_id)
            return None

        if self.agent_name == "Neuro_Symbolic_Auditor" and not self._auditor_should_intercept(incoming):
            logger.info("Auditor observed %s but no safety trigger found; no response", incoming.event_id)
            return None

        existing = store.list_events(incoming.case_id)
        if incoming.event_id not in {e.get("event_id") for e in existing}:
            store.append_event(incoming)
            existing = store.list_events(incoming.case_id)

        shared = merge_context_from_events(existing)
        shared.update(incoming.context_snapshot or {})
        case = self._load_case(incoming.case_id, incoming, shared)
        shared["case"] = case.to_dict()

        await self._emit_event(
            case_id=incoming.case_id,
            chat_id=chat_id,
            message_type="task",
            content=f"{self.agent_name} accepted Band handoff {incoming.event_id}",
            metadata={
                "protocol": "symbio.biosecurity.v1",
                "case_id": incoming.case_id,
                "incoming_event_id": incoming.event_id,
                "incoming_message_id": message_id,
                "agent": self.agent_name,
                "task_state": incoming.task_state,
                "raw_sender": (raw_message or {}).get("sender_name"),
            },
        )
        await self._emit_tool_call(case_id=incoming.case_id, chat_id=chat_id, incoming=incoming)

        try:
            envelope = await self.agent.run(case=case, events=existing, shared=shared)
        except Exception as exc: 
            logger.exception("%s runtime failed while processing %s", self.agent_name, incoming.event_id)
            await self._emit_event(
                case_id=incoming.case_id,
                chat_id=chat_id,
                message_type="error",
                content=f"{self.agent_name} failed before producing the next handoff: {exc.__class__.__name__}",
                metadata={
                    "protocol": "symbio.biosecurity.v1",
                    "case_id": incoming.case_id,
                    "failed_agent": self.agent_name,
                    "failed_event_id": incoming.event_id,
                    "error": str(exc)[:1000],
                    "error_type": exc.__class__.__name__,
                },
            )
            return await self._send_failure_handoff(incoming, case=case, exc=exc, chat_id=chat_id)

        envelope.context_snapshot = compact_shared_state(shared)
        envelope.context_snapshot.setdefault("case", case.to_dict())

        delivery = await self.transport.send_envelope(envelope)
        envelope.band_delivery = delivery.to_dict()
        store.append_event(envelope)
        store.save_case(case)
        await self._emit_tool_result(case_id=incoming.case_id, chat_id=chat_id, envelope=envelope)
        logger.info("%s processed %s and posted %s -> %s", self.agent_name, incoming.event_id, envelope.event_id, envelope.mention_targets())
        return envelope

    async def process_band_message(self, chat_id: str, payload: dict[str, Any]) -> SymbioEnvelope | None:
        message_id, content, source = band_message_fields(payload)
        incoming = envelope_from_band_content(content)
        if not incoming:
            logger.info("%s received non-Symbio message in %s; ignoring", self.agent_name, chat_id)
            if self.request_client and message_id:
                try:
                    await self.request_client.mark_processed(chat_id, message_id)
                except Exception:
                    logger.exception("Failed to mark ignored non-Symbio message processed")
            return None
        if self.request_client and message_id:
            await self.request_client.mark_processing(chat_id, message_id)
            await self.request_client.report_activity(chat_id, True)
        result: SymbioEnvelope | None = None
        try:
            result = await self.handle_envelope(incoming, chat_id=chat_id, message_id=message_id, raw_message=source)
        finally:
            if self.request_client:
                try:
                    await self.request_client.report_activity(chat_id, False)
                except Exception:
                    pass
        if self.request_client and message_id:
            if result and result.task_state == "agent_failed":
                await self.request_client.mark_failed(chat_id, message_id, result.payload.get("error", "Agent failed and failure handoff was posted."))
            else:
                await self.request_client.mark_processed(chat_id, message_id)
        return result

    async def run_forever(self) -> None:
        if not self.api_key:
            raise RuntimeError(f"Missing Band API key for {self.agent_name}; set BAND_AGENT_KEYS_JSON")
        if not self.request_client:
            raise RuntimeError("Request client not initialized")

        me = await self.request_client.get_me()
        logger.info("%s connected as Band profile: %s", self.agent_name, me)
        if not self.agent_id:
            self.agent_id = _extract_id(me) or ""

        chats = await self.request_client.list_chats()
        chat_ids = {str(c.get("id")) for c in chats if c.get("id")}
        if settings.band_case_room_id:
            chat_ids.add(settings.band_case_room_id)

        for chat_id in sorted(chat_ids):
            await self._drain_startup_queue(chat_id)

        socket = PhoenixChannelClient(api_key=self.api_key, agent_id=self.agent_id or None)
        await socket.connect()
        try:
            if self.agent_id:
                await socket.join(f"agent_rooms:{self.agent_id}")
            for chat_id in sorted(chat_ids):
                await socket.join(f"chat_room:{chat_id}")
                self.joined_chats.add(chat_id)
            async for event in socket.events():
                await self._handle_socket_event(socket, event)
        finally:
            await socket.close()

    async def _drain_startup_queue(self, chat_id: str) -> None:
        if not self.request_client:
            return
        while True:
            msg = await self.request_client.get_next_message(chat_id)
            if not msg:
                break
            await self.process_band_message(chat_id, msg)

    async def _handle_socket_event(self, socket: PhoenixChannelClient, event: PhoenixEvent) -> None:
        if event.event in {"phx_reply", "heartbeat"}:
            return
        if event.topic.startswith("agent_rooms:") and event.event == "room_added":
            room = event.payload.get("room") or event.payload.get("chat") or event.payload
            chat_id = str(room.get("id") or room.get("chat_id") or "")
            if chat_id and chat_id not in self.joined_chats:
                await socket.join(f"chat_room:{chat_id}")
                self.joined_chats.add(chat_id)
                await self._drain_startup_queue(chat_id)
            return
        if event.topic.startswith("agent_rooms:") and event.event == "room_removed":
            room = event.payload.get("room") or event.payload.get("chat") or event.payload
            chat_id = str(room.get("id") or room.get("chat_id") or "")
            if chat_id and chat_id in self.joined_chats:
                await socket.leave(f"chat_room:{chat_id}")
                self.joined_chats.remove(chat_id)
            return
        if event.topic.startswith("chat_room:") and event.event == "message_created":
            chat_id = event.topic.split(":", 1)[1]
            await self.process_band_message(chat_id, event.payload)

    async def _emit_event(self, *, case_id: str, chat_id: str | None, message_type: BandEventType, content: str, metadata: dict[str, Any] | None = None) -> None:
        if not settings.band_emit_events:
            return
        try:
            if chat_id and self.request_client and settings.band_enabled:
                await self.request_client.create_event(chat_id, message_type=message_type, content=content, metadata=metadata or {})
                store.append_band_record(case_id, {
                    "chat_id": chat_id, "sender": self.agent_name, "message_type": message_type,
                    "content": content, "metadata": metadata or {}, "mode": "band_event", "status": "sent",
                })
            else:
                await self.transport.record_event(case_id=case_id, agent=self.agent_name, message_type=message_type, content=content, metadata=metadata)
        except Exception:
            logger.exception("Failed to create Band semantic event")

    async def _emit_tool_call(self, *, case_id: str, chat_id: str | None, incoming: SymbioEnvelope) -> None:
        event_map: dict[str, tuple[BandEventType, str, dict[str, Any]]] = {
            "Vision_Analysis_Agent": ("tool_call", "Calling AI/ML API vision endpoint for neural crop perception", {"tool": "AI/ML API", "provider_role": "multimodal_vision"}),
            "Rule_Compliance_Agent": ("tool_call", "Running deterministic symbolic compliance rules", {"tool": "local_symbolic_rule_engine", "provider_role": "symbolic_guardrail"}),
            "Neuro_Symbolic_Auditor": ("tool_call", "Calling Featherless open-source auditor model", {"tool": "Featherless AI", "provider_role": "open_source_reasoning"}),
            "Supervisor_Agent": ("tool_call", "Calling Featherless supervisor report model", {"tool": "Featherless AI", "provider_role": "resilience_supervisor"}),
            "Operations_Report_Agent": ("tool_call", "Executing LangGraph report graph", {"tool": "LangGraph StateGraph", "provider_role": "heterogeneous_framework_agent"}),
        }
        spec = event_map.get(self.agent_name)
        if not spec:
            return
        message_type, content, metadata = spec
        await self._emit_event(case_id=case_id, chat_id=chat_id, message_type=message_type, content=content, metadata={**metadata, "case_id": case_id, "incoming_event_id": incoming.event_id})

    async def _emit_tool_result(self, *, case_id: str, chat_id: str | None, envelope: SymbioEnvelope) -> None:
        if self.agent_name == "Field_Intake_Agent":
            return
        metadata = {
            "protocol": "symbio.biosecurity.v1", "case_id": case_id, "agent": self.agent_name,
            "event_id": envelope.event_id, "task_state": envelope.task_state,
            "risk_level": envelope.risk_level, "requires_human_review": envelope.requires_human_review,
            "next_agent": envelope.next_agent,
        }
        payload = envelope.payload if isinstance(envelope.payload, dict) else {}
        if self.agent_name == "Rule_Compliance_Agent":
            metadata["triggered_rules"] = payload.get("triggered_rules", [])
            content = f"Symbolic rule check completed: {len(payload.get('triggered_rules', []))} rule(s) triggered."
        elif self.agent_name == "Operations_Report_Agent":
            metadata["framework"] = payload.get("framework")
            metadata["framework_trace"] = payload.get("framework_trace")
            content = f"Operations report graph completed via {payload.get('framework', 'unknown framework')}."
        else:
            content = f"{self.agent_name} completed tool/reasoning step and posted next handoff."
        await self._emit_event(case_id=case_id, chat_id=chat_id, message_type="tool_result", content=content, metadata=metadata)

    async def _send_failure_handoff(self, incoming: SymbioEnvelope, *, case: CropCase, exc: Exception, chat_id: str | None) -> SymbioEnvelope:
        failure_payload = {
            "failed_agent": self.agent_name, "failed_event_id": incoming.event_id,
            "error": str(exc)[:1000], "error_type": exc.__class__.__name__,
            "last_task_state": incoming.task_state, "expected_next_agent": incoming.next_agent,
            "reason": "Worker failed before producing a downstream Band handoff.",
        }
        next_agent = "Supervisor_Agent" if self.agent_name != "Supervisor_Agent" else "Human_Reviewer"
        observer_agents = ["Human_Reviewer"] if next_agent == "Supervisor_Agent" else []
        failure_envelope = SymbioEnvelope(
            case_id=incoming.case_id, agent=self.agent_name,
            role=f"{self.role if hasattr(self, 'role') else self.agent_name}_failure_boundary",
            task_state="agent_failed",
            finding=f"{self.agent_name} failed; posted resilience handoff to {next_agent} instead of leaving the Band room stalled.",
            next_agent=next_agent, observer_agents=observer_agents,
            handoff_reason="Self-healing failure path prevents a silent deadlock in the Band workflow.",
            input_refs=[incoming.event_id], confidence=0.0, risk_level="high", requires_human_review=True,
            payload=failure_payload,
            context_snapshot=compact_shared_state({**(incoming.context_snapshot or {}), "case": case.to_dict(), "failure": failure_payload}),
        )
        delivery = await self.transport.send_envelope(failure_envelope)
        failure_envelope.band_delivery = delivery.to_dict()
        store.append_event(failure_envelope)
        case.status = "agent_failed"
        case.requires_human_review = True
        case.risk_level = "high"
        store.save_case(case)
        return failure_envelope

    @staticmethod
    def _load_case(case_id: str, incoming: SymbioEnvelope, shared: dict[str, Any]) -> CropCase:
        try:
            return case_from_shared_or_store(case_id, shared)
        except KeyError:
            case_data = incoming.payload.get("case") if isinstance(incoming.payload, dict) else None
            if isinstance(case_data, dict):
                return CropCase.from_dict(case_data)
            raise

    @staticmethod
    def _auditor_should_intercept(incoming: SymbioEnvelope) -> bool:
        payload = incoming.payload or {}
        if incoming.agent == "Rule_Compliance_Agent" and payload.get("triggered_rules"):
            return True
        return False

def _extract_id(me_response: dict[str, Any]) -> str | None:
    data = me_response.get("data", me_response)
    if isinstance(data, dict):
        return data.get("id") or data.get("agent_id") or data.get("uuid")
    return None

class LocalBandMeshSimulator:
    def __init__(self) -> None:
        self.workers = {name: AgentWorker(name) for name in AGENT_FACTORIES}

    async def run_until_idle(self, initial_envelope: SymbioEnvelope, *, max_steps: int = 20) -> dict[str, Any]:
        transport = get_band_transport()
        initial_delivery = await transport.send_envelope(initial_envelope)
        initial_envelope.band_delivery = initial_delivery.to_dict()
        store.append_event(initial_envelope)
        queue: list[SymbioEnvelope] = [initial_envelope]
        steps = 0
        while queue and steps < max_steps:
            incoming = queue.pop(0)
            steps += 1
            targets = [t for t in incoming.mention_targets() if t in self.workers]
            for target in targets:
                result = await self.workers[target].handle_envelope(incoming)
                if result is not None and result.task_state not in TERMINAL_STATES:
                    queue.append(result)
                elif result is not None and result.mention_targets():
                    for t in result.mention_targets():
                        if t in self.workers:
                            queue.append(result)
        return {
            "events": store.list_events(initial_envelope.case_id),
            "band_transcript": store.list_band_records(initial_envelope.case_id),
            "steps": steps,
        }