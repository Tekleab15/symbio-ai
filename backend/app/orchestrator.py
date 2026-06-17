from __future__ import annotations

from typing import Any

from .agent_runtime import CONDUCTOR_AGENT, LocalBandMeshSimulator
from .models import SymbioEnvelope
from .services.band import get_band_transport
from .settings import settings
from .storage import store


class SymbioOrchestrator:
    def __init__(self) -> None:
        self.transport = get_band_transport()

    def initial_envelope_for_case(self, case_id: str) -> SymbioEnvelope:
        case = store.get_case(case_id)
        return SymbioEnvelope(
            case_id=case.case_id,
            agent=CONDUCTOR_AGENT,
            role="case_bootstrap",
            task_state="case_room_initialized",
            finding="New Symbio.AI biosecurity incident room initialized. Field intake agent owns the first task.",
            next_agent="Field_Intake_Agent",
            handoff_reason="Field intake must normalize the field report before any model or rule agent acts.",
            confidence=1.0,
            risk_level="unknown",
            requires_human_review=False,
            payload={"case": case.to_dict()},
            context_snapshot={"case": case.to_dict()},
        )

    async def dispatch_case(self, case_id: str) -> dict[str, Any]:
        case = store.get_case(case_id)
        case.status = "dispatched_to_band"
        store.save_case(case)
        initial = self.initial_envelope_for_case(case_id)

        if settings.mock_mode or not settings.band_enabled:
            result = await LocalBandMeshSimulator().run_until_idle(initial)
            return {"case": store.get_case(case_id).to_dict(), **result, "mode": "mock_event_mesh"}

        delivery = await self.transport.send_envelope(initial)
        initial.band_delivery = delivery.to_dict()
        store.append_event(initial)
        return {
            "case": case.to_dict(),
            "events": store.list_events(case_id),
            "band_transcript": store.list_band_records(case_id),
            "mode": "band_dispatched_only",
            "note": "Downstream execution now happens only when independent Band agent workers receive WebSocket @mentions.",
        }

    async def run_case(self, case_id: str) -> dict[str, Any]:
        return await self.dispatch_case(case_id)


orchestrator = SymbioOrchestrator()