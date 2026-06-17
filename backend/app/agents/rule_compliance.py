from __future__ import annotations

from typing import Any

from ..models import CropCase, SymbioEnvelope
from ..services.rules import rule_engine
from .base import AgentBase


class RuleComplianceAgent(AgentBase):
    name = "Rule_Compliance_Agent"
    role = "symbolic_compliance_gate"

    async def run(self, *, case: CropCase, events: list[dict[str, Any]], shared: dict[str, Any]) -> SymbioEnvelope:
        result = rule_engine.evaluate(case=case.to_dict(), vision=shared.get("vision", {}), agronomy=shared.get("agronomy", {}))
        shared["rules"] = result
        triggered = result.get("triggered_rules", [])
        next_agent = "Neuro_Symbolic_Auditor" if triggered else "Operations_Report_Agent"
        finding = "No deterministic rule violations." if not triggered else f"Triggered {len(triggered)} deterministic safety rule(s)."
        
        return SymbioEnvelope(
            case_id=case.case_id,
            agent=self.name,
            role=self.role,
            task_state="symbolic_gate_complete",
            finding=finding,
            confidence=float(result.get("symbolic_score", 0.0)),
            risk_level=result.get("risk_level", "unknown"),
            requires_human_review=bool(result.get("requires_human_review")),
            next_agent=next_agent,
            handoff_reason="Auditor must explain and challenge unsafe neural output." if triggered else "No violation; proceed to report and operations packet.",
            input_refs=self.last_event_ids(events, 3),
            payload=result,
        )