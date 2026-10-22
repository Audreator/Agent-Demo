"""Probe the DeepSeek native web search contract before relying on it.

Usage:
    python scripts/probe_search.py "杭州今天天气"

It prints the raw HTTP status and parsed links so the exact request shape can be
verified against the provider. Requires DEEPSEEK_SEARCH_API_KEY (or
MODEL_L1_API_KEY) in the environment.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings, reset_settings_cache  # noqa: E402
from app.models.search import DeepSeekWebSearch, SearchError  # noqa: E402


async def main(query: str) -> int:
    key = (
        os.environ.get("DEEPSEEK_SEARCH_API_KEY")
        or os.environ.get("MODEL_L1_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY")
        or ""
    )
    if not key:
        print("缺少 DEEPSEEK_SEARCH_API_KEY / MODEL_L1_API_KEY", file=sys.stderr)
        return 2
    reset_settings_cache()
    settings = Settings()
    provider = DeepSeekWebSearch(
        base_url=settings.deepseek_search_base_url,
        api_key=key,
        model=settings.deepseek_search_model,
    )
    try:
        results = await provider.search(query, max_results=5)
    except SearchError as exc:
        print(f"search failed: {exc}", file=sys.stderr)
        return 1
    print(f"results={len(results)}")
    for item in results:
        print(f"- {item['title']} | {item['url']}")
        print(f"  {item['snippet'][:160]}")
    return 0


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "杭州今天天气"
    raise SystemExit(asyncio.run(main(question)))
