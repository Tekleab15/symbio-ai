from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _json_env(name: str, default: Any) -> Any:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


@dataclass(frozen=True)
class Settings:
    mock_mode: bool = field(default_factory=lambda: _bool("MOCK_MODE", True))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("SYMBIO_DATA_DIR", "backend/runtime")))
    upload_dir: Path = field(default_factory=lambda: Path(os.getenv("UPLOAD_DIR", "backend/uploads")))
    public_base_url: str = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000")
    cors_origins: list[str] = field(default_factory=lambda: [x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if x.strip()])

    aimlapi_api_key: str = os.getenv("AIMLAPI_API_KEY", "")
    aimlapi_base_url: str = os.getenv("AIMLAPI_BASE_URL", "https://api.aimlapi.com/v1")
    aimlapi_vision_model: str = os.getenv("AIMLAPI_VISION_MODEL", "openai/gpt-4o-mini")

    featherless_api_key: str = os.getenv("FEATHERLESS_API_KEY", "")
    featherless_base_url: str = os.getenv("FEATHERLESS_BASE_URL", "https://api.featherless.ai/v1")
    featherless_model: str = os.getenv("FEATHERLESS_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    app_public_url: str = os.getenv("APP_PUBLIC_URL", "http://localhost:5173")
    app_title: str = os.getenv("APP_TITLE", "Symbio.AI Biosecurity Command")

    band_enabled: bool = field(default_factory=lambda: _bool("BAND_ENABLED", False))
    band_agent_api_base: str = os.getenv("BAND_AGENT_API_BASE", "https://app.band.ai/api/v1/agent")
    band_ws_url: str = os.getenv("BAND_WS_URL", "wss://app.band.ai/api/v1/socket/websocket")
    band_case_room_id: str = os.getenv("BAND_CASE_ROOM_ID", "")
    band_agent_keys: dict[str, str] = field(default_factory=lambda: _json_env("BAND_AGENT_KEYS_JSON", {}))
    band_agent_ids: dict[str, str] = field(default_factory=lambda: _json_env("BAND_AGENT_IDS_JSON", {}))
    band_participants: dict[str, dict[str, str]] = field(default_factory=lambda: _json_env("BAND_PARTICIPANTS_JSON", {}))
    band_ws_heartbeat_seconds: int = int(os.getenv("BAND_WS_HEARTBEAT_SECONDS", "25"))
    band_human_api_key: str = os.getenv("BAND_HUMAN_API_KEY", "")
    band_emit_events: bool = field(default_factory=lambda: _bool("BAND_EMIT_EVENTS", True))
    supervisor_enable_watchdog: bool = field(default_factory=lambda: _bool("SUPERVISOR_ENABLE_WATCHDOG", False))
    supervisor_timeout_seconds: int = int(os.getenv("SUPERVISOR_TIMEOUT_SECONDS", "60"))
    supervisor_scan_seconds: int = int(os.getenv("SUPERVISOR_SCAN_SECONDS", "5"))
    langgraph_force_fallback: bool = field(default_factory=lambda: _bool("LANGGRAPH_FORCE_FALLBACK", False))


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.upload_dir.mkdir(parents=True, exist_ok=True)