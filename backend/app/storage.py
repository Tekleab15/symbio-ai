from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from .models import CropCase, SymbioEnvelope
from .settings import settings


class JsonStore:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or settings.data_dir
        self.cases_dir = self.base_dir / "cases"
        self.events_dir = self.base_dir / "events"
        self.band_dir = self.base_dir / "band_transcripts"
        for directory in [self.cases_dir, self.events_dir, self.band_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _case_path(self, case_id: str) -> Path:
        return self.cases_dir / f"{case_id}.json"

    def _events_path(self, case_id: str) -> Path:
        return self.events_dir / f"{case_id}.json"

    def _band_path(self, case_id: str) -> Path:
        return self.band_dir / f"{case_id}.jsonl"

    def save_case(self, case: CropCase) -> None:
        with self._lock:
            self._case_path(case.case_id).write_text(json.dumps(case.to_dict(), indent=2), encoding="utf-8")

    def get_case(self, case_id: str) -> CropCase:
        path = self._case_path(case_id)
        if not path.exists():
            raise KeyError(f"Case not found: {case_id}")
        return CropCase.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_cases(self) -> list[dict[str, Any]]:
        cases = []
        for path in sorted(self.cases_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            cases.append(json.loads(path.read_text(encoding="utf-8")))
        return cases

    def clear_case_artifacts(self, case_id: str) -> None:
        with self._lock:
            for path in [self._events_path(case_id), self._band_path(case_id)]:
                if path.exists():
                    path.unlink()

    def append_event(self, envelope: SymbioEnvelope) -> None:
        with self._lock:
            events = self.list_events(envelope.case_id)
            events.append(envelope.to_dict())
            self._events_path(envelope.case_id).write_text(json.dumps(events, indent=2), encoding="utf-8")

    def list_events(self, case_id: str) -> list[dict[str, Any]]:
        path = self._events_path(case_id)
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def append_band_record(self, case_id: str, record: dict[str, Any]) -> None:
        with self._lock:
            with self._band_path(case_id).open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def list_band_records(self, case_id: str) -> list[dict[str, Any]]:
        path = self._band_path(case_id)
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

store = JsonStore()