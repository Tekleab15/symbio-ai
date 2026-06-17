from __future__ import annotations

import asyncio
import json
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()


async def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("Usage: python scripts/band_dump_participants.py <agent_api_key> <chat_room_id>")
    api_key = sys.argv[1]
    chat_id = sys.argv[2]
    base = os.getenv("BAND_AGENT_API_BASE", "https://app.band.ai/api/v1/agent").rstrip("/")
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(f"{base}/chats/{chat_id}/participants", headers={"X-API-Key": api_key})
        response.raise_for_status()
    print(json.dumps(response.json(), indent=2))
    print("\nPaste relevant participant objects into BAND_PARTICIPANTS_JSON in .env.")


if __name__ == "__main__":
    asyncio.run(main())