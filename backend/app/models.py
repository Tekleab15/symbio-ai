from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

RiskLevel = Literal["low", "medium", "high", "critical", "unknown"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:10]}"


@dataclass
class CropCase:
    case_id: str
    crop: str
    location: str
    symptoms: list[str]
    urgency: str = "medium"
    growth_stage: str = "unknown"
    acreage: float = 0.0
    image_path: str | None = None
    image_url: str | None = None
    created_at: str = field(default_factory=utc_now)
    status: str = "created"
    risk_level: RiskLevel = "unknown"
    requires_human_review: bool = False
    final_report: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CropCase":
        return cls(**data)


@dataclass
class SymbioEnvelope:
    """The structured state object exchanged through Band messages."""

    case_id: str
    agent: str
    role: str
    task_state: str
    finding: str
    next_agent: str | None = None
    handoff_reason: str | None = None
    observer_agents: list[str] = field(default_factory=list)
    input_refs: list[str] = field(default_factory=list)
    confidence: float | None = None
    risk_level: RiskLevel = "unknown"
    requires_human_review: bool = False
    payload: dict[str, Any] = field(default_factory=dict)
    context_snapshot: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: new_id("evt"))
    created_at: str = field(default_factory=utc_now)
    band_delivery: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SymbioEnvelope":
        allowed = {
            "case_id", "agent", "role", "task_state", "finding",
            "next_agent", "handoff_reason", "observer_agents",
            "input_refs", "confidence", "risk_level",
            "requires_human_review", "payload", "context_snapshot",
            "event_id", "created_at", "band_delivery"
        }
        return cls(**{k: v for k, v in data.items() if k in allowed})

    def mention_targets(self) -> list[str]:
        targets: list[str] = []
        if self.next_agent:
            targets.append(self.next_agent)
        targets.extend(self.observer_agents)
        seen: set[str] = set()
        ordered: list[str] = []
        for target in targets:
            if target and target not in seen:
                ordered.append(target)
                seen.add(target)
        return ordered

    def to_band_text(self) -> str:
        import json
        mentions = " ".join(f"@{target}" for target in self.mention_targets())
        mention_prefix = f"{mentions} " if mentions else ""
        body = {
            "protocol": "symbio.biosecurity.v1",
            "case_id": self.case_id,
            "event_id": self.event_id,
            "agent": self.agent,
            "role": self.role,
            "task_state": self.task_state,
            "input_refs": self.input_refs,
            "finding": self.finding,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "requires_human_review": self.requires_human_review,
            "next_agent": self.next_agent,
            "observer_agents": self.observer_agents,
            "handoff_reason": self.handoff_reason,
            "payload": self.payload,
            "context_snapshot": self.context_snapshot,
        }
        return f"{mention_prefix}```json\n{json.dumps(body, indent=2, ensure_ascii=False)}\n```"

@dataclass
class DiagnosisHypothesis:
    condition: str
    confidence: float
    category: str
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)