from __future__ import annotations

from typing import Any

from ..models import CropCase, SymbioEnvelope
from ..services.featherless import auditor_client
from .base import AgentBase


class NeuroSymbolicAuditorAgent(AgentBase):
    name = "Neuro_Symbolic_Auditor"
    role = "open_source_auditor"

    async def run(self, *, case: CropCase, events: list[dict[str, Any]], shared: dict[str, Any]) -> SymbioEnvelope:
        result = await auditor_client.explain(
            case=case.to_dict(),
            vision=shared.get("vision", {}),
            agronomy=shared.get("agronomy", {}),
            rules=shared.get("rules", {}),
        )
        shared["audit"] = result
        requires_review = bool(shared.get("rules", {}).get("requires_human_review"))
        finding = result.get("audit_summary", "Auditor completed review.")
        
        return SymbioEnvelope(
            case_id=case.case_id,
            agent=self.name,
            role=self.role,
            task_state="audit_complete",
            finding=finding,
            confidence=shared.get("rules", {}).get("symbolic_score", 0.0),
            risk_level=shared.get("rules", {}).get("risk_level", "unknown"),
            requires_human_review=requires_review,
            next_agent="Operations_Report_Agent",
            handoff_reason="Report agent must create final human-reviewable incident packet with blocked and allowed actions.",
            input_refs=self.last_event_ids(events, 3),
            payload=result,
        )