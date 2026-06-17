from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "agronomy_kb.json"


class AgronomyKnowledgeBase:
    def __init__(self, path: Path = DATA_PATH) -> None:
        self.path = path
        self.entries: list[dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))

    def match(self, crop: str, symptoms: list[str], hypotheses: list[dict[str, Any]]) -> dict[str, Any]:
        crop_l = crop.lower()
        symptom_text = " ".join(symptoms).lower()
        hypothesis_names = {h["condition"].lower() for h in hypotheses}
        scored: list[tuple[float, dict[str, Any]]] = []

        for entry in self.entries:
            score = 0.0
            if entry["condition"].lower() in hypothesis_names:
                score += 5.0
            if crop_l in [c.lower() for c in entry.get("crops", [])]:
                score += 2.0
            for kw in entry.get("symptom_keywords", []):
                if kw.lower() in symptom_text:
                    score += 1.0
            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [entry for _, entry in scored[:3]]
        evidence = []
        recommended_actions = []
        possible_treatments = []
        unsafe_actions = []
        quarantine_recommended = False
        chemical_action_recommended = False

        for entry in selected:
            evidence.extend(entry.get("evidence", []))
            recommended_actions.extend(entry.get("recommended_actions", []))
            possible_treatments.extend(entry.get("possible_treatments", []))
            unsafe_actions.extend(entry.get("unsafe_actions", []))
            quarantine_recommended = quarantine_recommended or bool(entry.get("quarantine_recommended", False))
            chemical_action_recommended = chemical_action_recommended or bool(entry.get("chemical_action_recommended", False))

        return {
            "matched_entries": selected,
            "evidence": list(dict.fromkeys(evidence)),
            "recommended_actions": list(dict.fromkeys(recommended_actions)),
            "possible_treatments": list(dict.fromkeys(possible_treatments)),
            "unsafe_actions": list(dict.fromkeys(unsafe_actions)),
            "quarantine_recommended": quarantine_recommended,
            "chemical_action_recommended": chemical_action_recommended,
            "evidence_count": len(set(evidence)),
        }

kb = AgronomyKnowledgeBase()