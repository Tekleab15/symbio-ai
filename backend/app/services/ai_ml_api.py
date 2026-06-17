from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import httpx

from ..settings import settings


class AIMLVisionClient:
    def __init__(self) -> None:
        self.base_url = settings.aimlapi_base_url.rstrip("/")
        self.api_key = settings.aimlapi_api_key
        self.model = settings.aimlapi_vision_model

    async def diagnose(self, *, crop: str, location: str, symptoms: list[str], image_url: str | None, image_path: str | None) -> dict[str, Any]:
        if settings.mock_mode or not self.api_key:
            return self.mock_diagnose(crop=crop, location=location, symptoms=symptoms)

        prompt = (
            "You are a cautious crop pathology triage agent. Return strict JSON only with keys: "
            "top_hypotheses (array of {condition, confidence, category, evidence}), "
            "visual_evidence (array), uncertainty, and recommended_next_observation. "
            "Do not recommend chemical treatment. Crop: "
            f"{crop}. Location: {location}. Symptoms: {', '.join(symptoms)}."
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        img = image_url or self._data_url_from_path(image_path)
        if img:
            content.append({"type": "image_url", "image_url": {"url": img}})

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.1,
            "max_tokens": 700,
        }
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
            )
            response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"]
        return self._parse_json_or_fallback(text, crop=crop, location=location, symptoms=symptoms)

    @staticmethod
    def _data_url_from_path(image_path: str | None) -> str | None:
        if not image_path:
            return None
        path = Path(image_path)
        if not path.exists():
            return None
        mime = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _parse_json_or_fallback(self, text: str, *, crop: str, location: str, symptoms: list[str]) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json\n", "", 1)
        try:
            data = json.loads(cleaned)
            if "top_hypotheses" in data:
                data["provider"] = "aimlapi"
                data["model"] = self.model
                return data
        except json.JSONDecodeError:
            pass
        fallback = self.mock_diagnose(crop=crop, location=location, symptoms=symptoms)
        fallback["provider"] = "aimlapi_unparseable_fallback"
        fallback["raw_model_text"] = text[:1000]
        return fallback

    @staticmethod
    def mock_diagnose(*, crop: str, location: str, symptoms: list[str]) -> dict[str, Any]:
        text = " ".join([crop, location, *symptoms]).lower()
        if "cassava" in text and ("streak" in text or "yellow" in text or "brown" in text):
            hyps = [
                {"condition": "Cassava Brown Streak Disease", "confidence": 0.72, "category": "viral", "evidence": ["yellowing", "brown streak-like pattern", "regional plausibility"]},
                {"condition": "nutrient deficiency", "confidence": 0.18, "category": "abiotic", "evidence": ["yellowing overlaps with nutrient stress"]},
                {"condition": "pest stress", "confidence": 0.10, "category": "pest", "evidence": ["non-specific leaf discoloration"]},
            ]
            uncertainty = "Moderate image quality and overlap with nutrient stress."
        elif "cactus" in text or "fig" in text or "cochineal" in text:
            hyps = [
                {"condition": "Cochineal scale infestation", "confidence": 0.88, "category": "pest", "evidence": ["cotton-like clusters", "cactus fig host", "localized patches"]},
                {"condition": "fungal lesion", "confidence": 0.08, "category": "fungal", "evidence": ["some discoloration"]},
                {"condition": "sunscald", "confidence": 0.04, "category": "abiotic", "evidence": ["surface discoloration"]},
            ]
            uncertainty = "Strong host/pest match; still requires field inspection."
        elif "tomato" in text:
            hyps = [
                {"condition": "Early blight", "confidence": 0.78, "category": "fungal", "evidence": ["brown spots", "leaf yellowing", "tomato host"]},
                {"condition": "bacterial spot", "confidence": 0.15, "category": "bacterial", "evidence": ["small lesions could overlap"]},
                {"condition": "nutrient deficiency", "confidence": 0.07, "category": "abiotic", "evidence": ["yellowing"]},
            ]
            uncertainty = "Visual symptoms are compatible with multiple leaf spot conditions."
        else:
            hyps = [
                {"condition": "unknown fungal or pest stress", "confidence": 0.55, "category": "unknown", "evidence": ["generic lesions", "insufficient crop-specific match"]},
                {"condition": "nutrient deficiency", "confidence": 0.25, "category": "abiotic", "evidence": ["discoloration"]},
                {"condition": "mechanical damage", "confidence": 0.20, "category": "abiotic", "evidence": ["non-specific injury"]},
            ]
            uncertainty = "Insufficient case details; request more images and field context."

        return {
            "provider": "mock",
            "model": "deterministic-demo-vision",
            "top_hypotheses": hyps,
            "visual_evidence": sorted({e for h in hyps for e in h.get("evidence", [])}),
            "uncertainty": uncertainty,
            "recommended_next_observation": "Collect a close-up image, whole-plant image, and spread-rate notes.",
        }

vision_client = AIMLVisionClient()