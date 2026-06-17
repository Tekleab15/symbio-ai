from __future__ import annotations

import json
import re
from typing import Any

from .models import CropCase, SymbioEnvelope
from .storage import store

_JSON_BLOCK = re.compile(r"```json\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_payload(content: str) -> dict[str, Any] | None:
    match = _JSON_BLOCK.search(content or "")
    candidate = match.group(1) if match else content
    candidate = candidate.strip()
    if not candidate:
        return None
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def envelope_from_band_content(content: str) -> SymbioEnvelope | None:
    data = extract_json_payload(content)
    if not data or not data.get("case_id"):
        return None
    try:
        return SymbioEnvelope.from_dict(data)
    except TypeError:
        return None


def compact_shared_state(shared: dict[str, Any]) -> dict[str, Any]:
    allowed = ["case", "intake", "vision", "agronomy", "rules", "audit", "report", "failure", "supervisor", "operation_graph"]
    return {key: shared[key] for key in allowed if key in shared}


def merge_context_from_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    shared: dict[str, Any] = {}
    for event in events:
        snapshot = event.get("context_snapshot") or {}
        if isinstance(snapshot, dict):
            shared.update(snapshot)
        agent = event.get("agent")
        payload = event.get("payload") or {}
        if not isinstance(payload, dict):
            continue
        if agent == "Field_Intake_Agent":
            shared["intake"] = payload
        elif agent == "Vision_Analysis_Agent":
            shared["vision"] = payload
        elif agent == "Agronomy_Knowledge_Agent":
            shared["agronomy"] = payload
        elif agent == "Rule_Compliance_Agent":
            shared["rules"] = payload
        elif agent == "Neuro_Symbolic_Auditor":
            shared["audit"] = payload
        elif agent == "Operations_Report_Agent":
            shared["report"] = payload
        elif agent == "Supervisor_Agent" or event.get("task_state") in {"agent_failed", "dead_drop_detected", "supervisor_escalated"}:
            shared["supervisor"] = payload
            if event.get("task_state") == "agent_failed":
                shared["failure"] = payload
    return shared


def case_from_shared_or_store(case_id: str, shared: dict[str, Any]) -> CropCase:
    case_data = shared.get("case")
    if isinstance(case_data, dict):
        try:
            return CropCase.from_dict(case_data)
        except TypeError:
            pass
    return store.get_case(case_id)


def band_message_fields(payload: dict[str, Any]) -> tuple[str | None, str, dict[str, Any]]:
    source = payload
    if isinstance(payload.get("message"), dict):
        source = payload["message"]
    elif isinstance(payload.get("data"), dict):
        source = payload["data"]
    message_id = source.get("id") or source.get("message_id")
    content = source.get("content") or source.get("text") or ""
    return message_id, content, source