from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.supervisor_runtime import RoomWatchdogSupervisor  # noqa: E402


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    await RoomWatchdogSupervisor().run_forever()


if __name__ == "__main__":
    asyncio.run(main())