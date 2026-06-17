from __future__ import annotations

from typing import Any

from ..models import CropCase, SymbioEnvelope
from ..services.knowledge import kb
from .base import AgentBase


class AgronomyKnowledgeAgent(AgentBase):
    name = "Agronomy_Knowledge_Agent"
    role = "agronomy_evidence"

    async def run(self, *, case: CropCase, events: list[dict[str, Any]], shared: dict[str, Any]) -> SymbioEnvelope:
        vision = shared.get("vision", {})
        result = kb.match(case.crop, case.symptoms, vision.get("top_hypotheses", []))
        shared["agronomy"] = result
        evidence_count = result.get("evidence_count", 0)
        finding = f"Matched {len(result.get('matched_entries', []))} agronomy entries with {evidence_count} evidence signals."
        
        return SymbioEnvelope(
            case_id=case.case_id,
            agent=self.name,
            role=self.role,
            task_state="evidence_attached",
            finding=finding,
            confidence=min(1.0, 0.35 + evidence_count * 0.15),
            risk_level="medium" if result.get("chemical_action_recommended") else "low",
            requires_human_review=False,
            next_agent="Rule_Compliance_Agent",
            observer_agents=["Neuro_Symbolic_Auditor"] if result.get("chemical_action_recommended") else [],
            handoff_reason="Symbolic compliance agent must verify confidence, evidence, quarantine, and chemical-safety constraints. Auditor is shadow-mentioned when chemical action appears.",
            input_refs=self.last_event_ids(events, 2),
            payload=result,
        )