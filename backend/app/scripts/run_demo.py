from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.main import run_demo_case  # noqa: E402


async def main() -> None:
    case_id = sys.argv[1] if len(sys.argv) > 1 else "cassava-low-confidence"
    result = await run_demo_case(case_id)
    print(json.dumps(result["case"], indent=2))
    print("\nEvents:")
    for event in result["events"]:
        print(f"- {event['agent']} -> {event.get('next_agent')}: {event['task_state']} [{event['risk_level']}]")
    print("\nReport:")
    print(json.dumps(result["case"].get("final_report", {}), indent=2))


if __name__ == "__main__":
    asyncio.run(main())