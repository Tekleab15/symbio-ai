from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, AsyncIterator
from urllib.parse import urlencode

import websockets
from websockets.client import WebSocketClientProtocol

from ..settings import settings

@dataclass
class PhoenixEvent:
    join_ref: str | None
    ref: str | None
    topic: str
    event: str
    payload: dict[str, Any]

class PhoenixChannelClient:
    """Minimal Phoenix Channels client for Band WebSocket subscriptions."""

    def __init__(self, *, api_key: str, agent_id: str | None = None, ws_url: str | None = None) -> None:
        self.api_key = api_key
        self.agent_id = agent_id
        self.ws_url = ws_url or settings.band_ws_url
        self._ref = 0
        self._join_ref = 0
        self._ws: WebSocketClientProtocol | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None

    def _next_ref(self) -> str:
        self._ref += 1
        return str(self._ref)

    def _next_join_ref(self) -> str:
        self._join_ref += 1
        return str(self._join_ref)

    def connection_url(self) -> str:
        params = {"api_key": self.api_key, "vsn": "2.0.0"}
        if self.agent_id:
            params["agent_id"] = self.agent_id
        sep = "&" if "?" in self.ws_url else "?"
        return f"{self.ws_url}{sep}{urlencode(params)}"

    async def connect(self) -> None:
        self._ws = await websockets.connect(self.connection_url(), ping_interval=None)
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def close(self) -> None:
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._ws:
            await self._ws.close()

    async def join(self, topic: str) -> None:
        if not self._ws:
            raise RuntimeError("WebSocket not connected")
        join_ref = self._next_join_ref()
        ref = self._next_ref()
        await self._ws.send(json.dumps([join_ref, ref, topic, "phx_join", {}]))

    async def leave(self, topic: str) -> None:
        if not self._ws:
            return
        await self._ws.send(json.dumps([None, self._next_ref(), topic, "phx_leave", {}]))

    async def events(self) -> AsyncIterator[PhoenixEvent]:
        if not self._ws:
            raise RuntimeError("WebSocket not connected")
        async for raw in self._ws:
            try:
                join_ref, ref, topic, event, payload = json.loads(raw)
            except Exception:
                continue
            yield PhoenixEvent(join_ref=join_ref, ref=ref, topic=topic, event=event, payload=payload or {})

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(settings.band_ws_heartbeat_seconds)
            try:
                if self._ws:
                    await self._ws.send(json.dumps([None, self._next_ref(), "phoenix", "heartbeat", {}]))
            except Exception:
                return