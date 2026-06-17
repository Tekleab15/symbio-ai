from __future__ import annotations

import json
from typing import Any

import httpx

from ..settings import settings


class FeatherlessAuditorClient:
    def __init__(self) -> None:
        self.api_key = settings.featherless_api_key
        self.base_url = settings.featherless_base_url.rstrip("/")
        self.model = settings.featherless_model

    async def explain(self, *, case: dict[str, Any], vision: dict[str, Any], agronomy: dict[str, Any], rules: dict[str, Any]) -> dict[str, Any]:
        if settings.mock_mode or not self.api_key:
            return self.template_explanation(case=case, vision=vision, agronomy=agronomy, rules=rules)

        prompt = (
            "You are the Neuro_Symbolic_Auditor in Symbio.AI. A deterministic rule engine has already decided the safety gate. "
            "Your task is to explain the violation to other agents in a concise, strict, operational style. "
            "Return JSON with keys: audit_summary, correction_request, human_escalation_reason, blocked_actions, safe_next_steps. "
            "Do not override the deterministic rule engine.\n\n"
            f"CASE:\n{json.dumps(case, indent=2)}\n\n"
            f"VISION:\n{json.dumps(vision, indent=2)}\n\n"
            f"AGRONOMY:\n{json.dumps(agronomy, indent=2)}\n\n"
            f"RULES:\n{json.dumps(rules, indent=2)}"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You explain deterministic neuro-symbolic safety gates for high-stakes crop biosecurity workflows."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 500,
        }
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": settings.app_public_url,
                    "X-Title": settings.app_title,
                },
                json=payload,
            )
            response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        return self._parse_json_or_template(text, case=case, vision=vision, agronomy=agronomy, rules=rules)

    def _parse_json_or_template(self, text: str, *, case: dict[str, Any], vision: dict[str, Any], agronomy: dict[str, Any], rules: dict[str, Any]) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").replace("json\n", "", 1)
        try:
            data = json.loads(cleaned)
            data["provider"] = "featherless"
            data["model"] = self.model
            return data
        except json.JSONDecodeError:
            data = self.template_explanation(case=case, vision=vision, agronomy=agronomy, rules=rules)
            data["provider"] = "featherless_unparseable_fallback"
            data["raw_model_text"] = text[:1000]
            return data

    @staticmethod
    def template_explanation(*, case: dict[str, Any], vision: dict[str, Any], agronomy: dict[str, Any], rules: dict[str, Any]) -> dict[str, Any]:
        triggered = rules.get("triggered_rules", [])
        top = vision.get("top_hypotheses", [{}])[0]
        if triggered:
            lines = [f"{r['rule_id']}: {r['reason']}" for r in triggered]
            summary = "SYMBOLIC SAFETY INTERCEPT: " + " | ".join(lines)
        else:
            summary = "No symbolic safety violation found. Proceed with non-irreversible operational planning."
        return {
            "provider": "mock",
            "model": "deterministic-auditor-template",
            "audit_summary": summary,
            "correction_request": f"Top neural hypothesis is {top.get('condition', 'unknown')} at {top.get('confidence', 0):.2f}. Follow deterministic gate before intervention.",
            "human_escalation_reason": "Human agronomist review required." if rules.get("requires_human_review") else "No mandatory human escalation.",
            "blocked_actions": rules.get("blocked_actions", []),
            "safe_next_steps": rules.get("allowed_actions", []),
        }

    async def explain_supervisor_failure(self, *, case: dict[str, Any], failure: dict[str, Any], recent_events: list[dict[str, Any]]) -> dict[str, Any]:
        if settings.mock_mode or not self.api_key:
            return self.template_supervisor_failure(case=case, failure=failure, recent_events=recent_events)

        prompt = (
            "You are Supervisor_Agent for a Band-powered multi-agent biosecurity workflow. "
            "A worker failed or the room stalled. Return JSON with keys: supervisor_summary, "
            "suspected_stalled_agent, recovery_action, human_escalation_reason, retry_recommendation, recent_event_ids. "
            "Be concise and operational. Do not claim the biological diagnosis is certain.\n\n"
            f"CASE:\n{json.dumps(case, indent=2)}\n\n"
            f"FAILURE:\n{json.dumps(failure, indent=2)}\n\n"
            f"RECENT_EVENTS:\n{json.dumps(recent_events[-5:], indent=2)}"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You summarize distributed agent workflow failures for human operators."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 500,
        }
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": settings.app_public_url,
                    "X-Title": settings.app_title,
                },
                json=payload,
            )
            response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").replace("json\n", "", 1)
        try:
            data = json.loads(cleaned)
            data["provider"] = "featherless"
            data["model"] = self.model
            return data
        except json.JSONDecodeError:
            data = self.template_supervisor_failure(case=case, failure=failure, recent_events=recent_events)
            data["provider"] = "featherless_unparseable_fallback"
            data["raw_model_text"] = text[:1000]
            return data

    @staticmethod
    def template_supervisor_failure(*, case: dict[str, Any], failure: dict[str, Any], recent_events: list[dict[str, Any]]) -> dict[str, Any]:
        failed_agent = failure.get("failed_agent") or failure.get("stalled_after_agent") or "unknown_agent"
        return {
            "provider": "mock",
            "model": "deterministic-supervisor-template",
            "supervisor_summary": f"MESH RESILIENCE ALERT: workflow stalled or failed around {failed_agent} for case {case.get('case_id', 'unknown')}.",
            "suspected_stalled_agent": failed_agent,
            "recovery_action": "Escalate to human reviewer and preserve current Band context before retrying the failed handoff.",
            "human_escalation_reason": failure.get("error") or failure.get("reason") or "No downstream handoff was observed before the timeout window.",
            "retry_recommendation": "Retry from the last successful SymbioEnvelope after checking API keys, model endpoint latency, and participant metadata.",
            "recent_event_ids": [e.get("event_id") for e in recent_events[-5:]],
        }

auditor_client = FeatherlessAuditorClient()