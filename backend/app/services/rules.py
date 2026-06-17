from __future__ import annotations

from typing import Any


def top_confidence(vision_payload: dict[str, Any]) -> float:
    hyps = vision_payload.get("top_hypotheses", [])
    if not hyps:
        return 0.0
    return float(max(h.get("confidence", 0.0) for h in hyps))


def second_confidence(vision_payload: dict[str, Any]) -> float:
    hyps = sorted(vision_payload.get("top_hypotheses", []), key=lambda h: h.get("confidence", 0.0), reverse=True)
    if len(hyps) < 2:
        return 0.0
    return float(hyps[1].get("confidence", 0.0))


class RuleEngine:
    """Deterministic safety gate.
    The LLM auditor may explain violations, but this engine owns the actual gate.
    """

    def evaluate(self, *, case: dict[str, Any], vision: dict[str, Any], agronomy: dict[str, Any]) -> dict[str, Any]:
        triggered: list[dict[str, Any]] = []
        max_conf = top_confidence(vision)
        second_conf = second_confidence(vision)
        evidence_count = int(agronomy.get("evidence_count", 0))
        urgency = case.get("urgency", "medium")
        acreage = float(case.get("acreage", 0.0) or 0.0)
        symptom_text = " ".join(case.get("symptoms", [])).lower()

        def add(rule_id: str, severity: str, decision: str, reason: str, action: str) -> None:
            triggered.append({
                "rule_id": rule_id,
                "severity": severity,
                "decision": decision,
                "reason": reason,
                "action": action,
            })

        if max_conf < 0.80:
            add("RISK-LOW-CONFIDENCE-001", "medium", "block_irreversible_action", f"Top vision confidence is {max_conf:.2f}, below 0.80.", "Request clearer image and human agronomist review before quarantine or chemical action.")

        if max_conf - second_conf < 0.20 and second_conf > 0:
            add("RISK-CLOSE-HYPOTHESES-002", "medium", "request_more_evidence", f"Top and second hypotheses are close: {max_conf:.2f} vs {second_conf:.2f}.", "Ask for field spread rate and second image before action.")

        if agronomy.get("chemical_action_recommended") and max_conf < 0.85:
            add("CHEMICAL-SAFETY-003", "high", "block_chemical_action", "Chemical intervention appears in possible treatments but diagnosis is not confirmed above 0.85.", "Permit inspection and sanitation only until human approval.")

        if agronomy.get("quarantine_recommended") and evidence_count < 2:
            add("BIOSEC-QUARANTINE-004", "high", "block_quarantine_order", "Quarantine requires at least two independent evidence signals.", "Prepare watch-zone advisory instead of quarantine.")

        if urgency == "high" and acreage >= 2.0 and any(word in symptom_text for word in ["spreading", "neighboring", "rapid", "widespread"]):
            add("SUPPLY-CHAIN-SPREAD-005", "high", "escalate_ops", "Large or spreading incident may affect supply-chain planning.", "Notify operations agent and prepare field visit within 24 hours.")

        blocked_actions = [r["decision"] for r in triggered if r["decision"].startswith("block")]
        requires_human_review = any(r["severity"] in {"high", "critical"} for r in triggered) or bool(blocked_actions)
        risk_order = {"unknown": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        risk_level = "low"
        for r in triggered:
            if risk_order[r["severity"]] > risk_order[risk_level]:
                risk_level = r["severity"]

        return {
            "triggered_rules": triggered,
            "requires_human_review": requires_human_review,
            "risk_level": risk_level,
            "blocked_actions": blocked_actions,
            "allowed_actions": self._allowed_actions(triggered),
            "symbolic_score": max(0.0, 1.0 - 0.18 * len(triggered)),
        }

    @staticmethod
    def _allowed_actions(triggered: list[dict[str, Any]]) -> list[str]:
        blocked = {r["decision"] for r in triggered}
        actions = ["collect_second_image", "inspect_neighboring_plants", "record_gps_boundary", "non_chemical_sanitation", "human_agronomist_review"]
        if "block_quarantine_order" not in blocked:
            actions.append("provisional_watch_zone")
        if "block_chemical_action" not in blocked:
            actions.append("expert_approved_treatment_planning")
        return actions

rule_engine = RuleEngine()