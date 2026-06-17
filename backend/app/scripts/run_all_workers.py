from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.agent_runtime import AGENT_FACTORIES, AgentWorker  # noqa: E402
from app.settings import settings  # noqa: E402
from app.supervisor_runtime import RoomWatchdogSupervisor  # noqa: E402


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    workers = [AgentWorker(name).run_forever() for name in AGENT_FACTORIES]
    if settings.supervisor_enable_watchdog:
        workers.append(RoomWatchdogSupervisor().run_forever())
    await asyncio.gather(*workers)


if __name__ == "__main__":
    asyncio.run(main())