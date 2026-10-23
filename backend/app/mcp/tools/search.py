from __future__ import annotations

import json
import os
from typing import Any

from app.citations import apply_link_statuses, check_links, merge_citations
from app.mcp.tools.artifacts import save_artifact
from app.models.search import DeepSeekWebSearch, SearchError, enforce_size_limit


async def search_tool(
    query: str,
    max_results: int = 5,
    recency_days: int | None = None,
    site: str | None = None,
    *,
    artifact_id: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    if not isinstance(max_results, int) or isinstance(max_results, bool):
        raise SearchError("max_results 必须是整数")
    provider = DeepSeekWebSearch()
    results = await provider.search(
        query, max_results=max_results, recency_days=recency_days, site=site
    )
    citation_limit = max(1, int(os.environ.get("CITATION_MAX_TOTAL", "10")))
    citations = merge_citations(
        [results], prefix=f"{artifact_id}-" if artifact_id else "", limit=citation_limit
    )
    if citations:
        statuses = await check_links([item["url"] for item in citations])
        apply_link_statuses(citations, statuses)

    limit = int(os.environ.get("SEARCH_RESULT_LIMIT", "8000"))
    kept, truncated = enforce_size_limit(results, max(1000, limit - 512))
    payload: dict[str, Any] = {
        "query": query,
        "source": "deepseek_web_search",
        "count": len(kept),
        "results": kept,
        "truncated": truncated,
        "citations": citations,
        "reachable_count": sum(1 for item in citations if item["reachable"]),
    }
    if not truncated and len(json.dumps(payload, ensure_ascii=False)) > limit:
        truncated = True
        payload["truncated"] = True
    if truncated:
        full = json.dumps(
            {"query": query, "results": results, "citations": citations},
            ensure_ascii=False,
            indent=2,
        )
        payload["artifact"] = save_artifact(
            full, artifact_id=artifact_id, session_id=session_id
        )
        payload["note"] = "结果超出长度上限，已截断并落 artifact，可用 read_docs 回读"
        while (
            len(json.dumps(payload, ensure_ascii=False)) > limit
            and payload["results"]
        ):
            payload["results"].pop()
            payload["count"] = len(payload["results"])
    return payload


__all__ = ["search_tool"]
