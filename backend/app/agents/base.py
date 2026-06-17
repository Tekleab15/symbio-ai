from __future__ import annotations

from typing import Any

from ..models import CropCase, SymbioEnvelope


class AgentBase:
    name: str = "Agent"
    role: str = "generic"

    async def run(self, *, case: CropCase, events: list[dict[str, Any]], shared: dict[str, Any]) -> SymbioEnvelope:
        raise NotImplementedError

    def last_event_ids(self, events: list[dict[str, Any]], n: int = 2) -> list[str]:
        return [e["event_id"] for e in events[-n:]]