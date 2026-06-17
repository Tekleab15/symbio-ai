from __future__ import annotations

from typing import Any

from ..models import CropCase, SymbioEnvelope
from .base import AgentBase


class FieldIntakeAgent(AgentBase):
    name = "Field_Intake_Agent"
    role = "field_intake"

    async def run(self, *, case: CropCase, events: list[dict[str, Any]], shared: dict[str, Any]) -> SymbioEnvelope:
        missing = []
        if not case.crop:
            missing.append("crop")
        if not case.location:
            missing.append("location")
        if not case.symptoms:
            missing.append("symptoms")
        if not case.image_path and not case.image_url:
            missing.append("image")

        completeness = max(0.0, 1.0 - len(missing) * 0.2)
        shared["intake"] = {
            "missing_fields": missing,
            "completeness": completeness,
            "normalized_symptoms": case.symptoms,
            "field_context": {
                "crop": case.crop,
                "location": case.location,
                "urgency": case.urgency,
                "growth_stage": case.growth_stage,
                "acreage": case.acreage,
            },
        }
        task_state = "needs_more_information" if missing else "ready_for_vision"
        next_agent = "Vision_Analysis_Agent" if not missing else "Operations_Report_Agent"
        finding = "Case intake complete; structured field report ready for neural vision triage." if not missing else f"Missing required fields: {', '.join(missing)}."
        
        return SymbioEnvelope(
            case_id=case.case_id,
            agent=self.name,
            role=self.role,
            task_state=task_state,
            finding=finding,
            confidence=completeness,
            risk_level="medium" if missing else "low",
            requires_human_review=bool(missing),
            next_agent=next_agent,
            handoff_reason="Vision agent needs normalized crop, location, symptoms, and image pointer." if not missing else "Report agent should ask human for missing information.",
            input_refs=[],
            payload=shared["intake"],
        )