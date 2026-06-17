from __future__ import annotations

from typing import Any

from ..models import CropCase, SymbioEnvelope
from ..services.ai_ml_api import vision_client
from .base import AgentBase


class VisionAnalysisAgent(AgentBase):
    name = "Vision_Analysis_Agent"
    role = "neural_vision_analysis"

    async def run(self, *, case: CropCase, events: list[dict[str, Any]], shared: dict[str, Any]) -> SymbioEnvelope:
        result = await vision_client.diagnose(
            crop=case.crop,
            location=case.location,
            symptoms=case.symptoms,
            image_url=case.image_url,
            image_path=case.image_path,
        )
        shared["vision"] = result
        top = result.get("top_hypotheses", [{}])[0]
        confidence = float(top.get("confidence", 0.0))
        finding = f"Top neural hypothesis: {top.get('condition', 'unknown')} at confidence {confidence:.2f}."
        
        return SymbioEnvelope(
            case_id=case.case_id,
            agent=self.name,
            role=self.role,
            task_state="vision_posted",
            finding=finding,
            confidence=confidence,
            risk_level="medium" if confidence < 0.80 else "low",
            requires_human_review=confidence < 0.65,
            next_agent="Agronomy_Knowledge_Agent",
            observer_agents=["Neuro_Symbolic_Auditor"] if confidence < 0.75 else [],
            handoff_reason="Agronomy agent must ground neural hypotheses against crop-specific evidence and safe response options. Auditor is shadow-mentioned for low-confidence neural output.",
            input_refs=self.last_event_ids(events, 1),
            payload=result,
        )