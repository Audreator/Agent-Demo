from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import urlparse

import httpx

from app.errors import TransientError


class SearchError(ValueError):
    pass


def _extract_responses_results(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse the Responses API output for web_search citations."""
    results: list[dict[str, Any]] = []
    texts: list[str] = []
    for item in data.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if not isinstance(part, dict):
                continue
            text = str(part.get("text") or "")
            if text:
                texts.append(text)
            for annotation in part.get("annotations") or []:
                if not isinstance(annotation, dict):
                    continue
                if annotation.get("type") != "url_citation":
                    continue
                citation = annotation.get("url_citation") or {}
                title = citation.get("title") or annotation.get("title")
                url = (
                    annotation.get("url")
                    or citation.get("url")
                    or citation.get("link")
                )
                if not title or not url:
                    continue
                if urlparse(str(url)).scheme not in {"http", "https"}:
                    continue
                snippet = ""
                start = annotation.get("start_index")
                end = annotation.get("end_index")
                if isinstance(start, int) and isinstance(end, int) and text:
                    snippet = text[max(0, start - 80) : min(len(text), end + 160)]
                results.append(
                    {"title": str(title), "url": str(url), "snippet": snippet}
                )
    return results


class DeepSeekWebSearch:
    """Search provider fixed to DeepSeek native web search, independent of the chat model layer."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        enabled: bool | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get("DEEPSEEK_SEARCH_BASE_URL")
            or os.environ.get("MODEL_L1_BASE_URL")
            or "https://api.deepseek.com"
        ).rstrip("/")
        self.api_key = (
            api_key
            or os.environ.get("DEEPSEEK_SEARCH_API_KEY")
            or os.environ.get("MODEL_L1_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY", "")
        )
        self.model = model or os.environ.get(
            "DEEPSEEK_SEARCH_MODEL", "deepseek-v4-pro"
        )
        env_enabled = os.environ.get("DEEPSEEK_SEARCH_ENABLED", "true").lower() != "false"
        self.enabled = env_enabled if enabled is None else enabled
        self.timeout = timeout

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        recency_days: int | None = None,
        site: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self.enabled:
            raise SearchError("联网搜索已禁用（DEEPSEEK_SEARCH_ENABLED=false）")
        if not self.api_key:
            raise SearchError("未配置 DEEPSEEK_SEARCH_API_KEY，无法执行联网搜索")
        if not query.strip():
            raise SearchError("query 不能为空")
        if not 1 <= max_results <= 10:
            raise SearchError("max_results 必须在 1-10 之间")
        if recency_days is not None and (recency_days < 1 or recency_days > 3650):
            raise SearchError("recency_days 必须在 1-3650 之间")

        results = await self._responses_search(
            query, max_results, recency_days, site
        )
        if not results:
            raise SearchError("搜索未返回可解析的结果，请更换关键词后重试")
        cleaned: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in results:
            url = item["url"]
            if url in seen:
                continue
            seen.add(url)
            cleaned.append(item)
            if len(cleaned) >= max_results:
                break
        if not cleaned:
            raise SearchError("搜索未返回有效链接")
        return cleaned

    async def _responses_search(
        self,
        query: str,
        max_results: int,
        recency_days: int | None,
        site: str | None,
    ) -> list[dict[str, Any]]:
        limitations = []
        if recency_days:
            limitations.append(f"只关注最近 {recency_days} 天的信息")
        if site:
            limitations.append(f"限定站点 {site}")
        instructions = "使用联网搜索回答问题，并基于搜索结果给出中文摘要。"
        if limitations:
            instructions += "；" + "；".join(limitations) + "。"
        payload: dict[str, Any] = {
            "model": self.model,
            "input": query,
            "instructions": instructions,
            "tools": [{"type": "web_search"}],
            "tool_choice": "auto",
            "stream": False,
        }
        data = await self._post("/responses", payload)
        results = _extract_responses_results(data)
        if not results:
            raise SearchError("联网搜索未返回 url_citation 结果，拒绝使用模型编造链接")
        return results

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}{path}", json=payload, headers=headers
                )
        except httpx.TimeoutException as exc:
            raise TransientError(f"搜索请求超时: {exc}") from exc
        except httpx.HTTPError as exc:
            raise TransientError(f"搜索请求失败: {exc}") from exc
        if response.status_code >= 400:
            raise SearchError(
                f"搜索接口返回 {response.status_code}: {response.text[:300]}"
            )
        data = response.json()
        return data if isinstance(data, dict) else {}


def enforce_size_limit(
    results: list[dict[str, Any]], limit: int
) -> tuple[list[dict[str, Any]], bool]:
    """Keep JSON payload under limit by trimming snippets then dropping tail results."""
    compact = [dict(item) for item in results]
    if len(json.dumps(compact, ensure_ascii=False)) <= limit:
        return compact, False
    truncated = True
    for item in compact:
        snippet = str(item.get("snippet", ""))
        if len(snippet) > 200:
            item["snippet"] = snippet[:200] + "…"
    while compact and len(json.dumps(compact, ensure_ascii=False)) > limit:
        compact.pop()
    return compact, truncated


__all__ = ["DeepSeekWebSearch", "SearchError", "enforce_size_limit"]
