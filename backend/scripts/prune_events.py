from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import delete  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.db import get_sessionmaker  # noqa: E402
from app.models.db_models import Event  # noqa: E402


async def prune(days: int) -> int:
    cutoff = datetime.now(UTC) - timedelta(days=days)
    sessionmaker = get_sessionmaker(get_settings())
    async with sessionmaker() as db:
        result = await db.execute(delete(Event).where(Event.ts < cutoff))
        await db.commit()
        return int(result.rowcount or 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="清理过期 events 记录")
    parser.add_argument(
        "--days",
        type=int,
        default=get_settings().events_retention_days,
        help="保留天数（默认读取 EVENTS_RETENTION_DAYS）",
    )
    args = parser.parse_args()
    if args.days < 1:
        parser.error("--days 必须大于 0")
    deleted = asyncio.run(prune(args.days))
    print(f"已清理 {deleted} 条早于 {args.days} 天的 events 记录")


if __name__ == "__main__":
    main()
