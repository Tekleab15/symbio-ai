from __future__ import annotations

from typing import Any

from ..models import CropCase, SymbioEnvelope
from ..services.featherless import auditor_client
from .base import AgentBase


class SupervisorAgent(AgentBase):
    name = "Supervisor_Agent"
    role = "mesh_resilience_supervisor"

    async def run(self, *, case: CropCase, events: list[dict[str, Any]], shared: dict[str, Any]) -> SymbioEnvelope:
        failure = shared.get("failure") or shared.get("supervisor") or self._infer_failure(events)
        result = await auditor_client.explain_supervisor_failure(
            case=case.to_dict(),
            failure=failure,
            recent_events=events,
        )
        shared["supervisor"] = result
        case.status = "supervisor_escalated"
        case.requires_human_review = True
        case.risk_level = "high"
        
        return SymbioEnvelope(
            case_id=case.case_id,
            agent=self.name,
            role=self.role,
            task_state="supervisor_escalated",
            finding=result.get("supervisor_summary", "Supervisor escalated the workflow to a human reviewer."),
            confidence=0.9,
            risk_level="high",
            requires_human_review=True,
            next_agent="Human_Reviewer",
            handoff_reason="A distributed-agent failure or dead drop requires human intervention before the workflow can continue.",
            input_refs=self.last_event_ids(events, 5),
            payload=result,
        )

    @staticmethod
    def _infer_failure(events: list[dict[str, Any]]) -> dict[str, Any]:
        last = events[-1] if events else {}
        return {
            "reason": "Supervisor was invoked without an explicit failure payload.",
            "stalled_after_agent": last.get("agent", "unknown_agent"),
            "last_event_id": last.get("event_id"),
            "last_task_state": last.get("task_state"),
            "expected_next_agent": last.get("next_agent"),
        }