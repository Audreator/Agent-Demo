from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest

from app.admin.service import percentile
from app.api.admin import _build_csv, _safe_cell
from app.citations import (
    LinkStatus,
    check_link,
    merge_citations,
    normalize_url,
    validate_url,
)
from app.mcp.tools.search import search_tool


def test_normalize_url() -> None:
    assert (
        normalize_url("HTTPS://Example.COM/Path?Q=1#frag")
        == "https://example.com/Path?Q=1"
    )
    assert normalize_url("ftp://example.com/a") == ""
    assert normalize_url("file:///etc/passwd") == ""
    assert normalize_url("https://example.com/" + "a" * 2100) == ""
    assert normalize_url("http://[::1") == ""


def test_merge_citations_dedupe_order_and_limit() -> None:
    groups = [
        [
            {"title": "A", "url": "https://Example.com/a#x", "snippet": "a" * 400},
            {"title": "B", "url": "https://example.com/b", "snippet": "b"},
            {"title": "A2", "url": "https://example.com/a", "snippet": "dup"},
        ],
        [{"title": "C", "url": "https://example.com/c", "snippet": "c"}],
    ]
    merged = merge_citations(groups, prefix="tc-", limit=2)
    assert len(merged) == 2
    assert [item["url"] for item in merged] == [
        "https://example.com/a",
        "https://example.com/b",
    ]
    assert merged[0]["id"] == "tc-c1"
    assert len(merged[0]["title"]) <= 200
    assert len(merged[0]["snippet"]) <= 300
    many = merge_citations(
        [[{"title": str(index), "url": f"https://example.com/{index}"} for index in range(12)]],
        limit=10,
    )
    assert len(many) == 10


async def test_validate_url_rejects_private_and_non_http() -> None:
    assert await validate_url("file:///etc/passwd") is not None
    assert await validate_url("http://127.0.0.1/a") is not None
    assert await validate_url("http://10.0.0.2/a") is not None
    assert await validate_url("http://localhost/a") is not None
    assert await validate_url("http://[::1]/a") is not None


async def test_validate_url_rejects_private_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_resolve(host: str) -> bool:
        return False

    monkeypatch.setattr("app.citations.link_check._resolve_host", fake_resolve)
    assert await validate_url("https://public.example.com/a") is not None


class _FakeResponse:
    def __init__(self, status_code: int, headers: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self.headers = headers or {}


class _FakeStream:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.status_code = response.status_code
        self.headers = response.headers

    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def aiter_bytes(self, chunk_size: int = 0) -> Any:
        yield b"x" * min(16, chunk_size or 16)


def _fake_client(
    head_handler: Any, get_handler: Any | None = None
) -> Any:
    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def head(self, url: str, **kwargs: Any) -> _FakeResponse:
            result = head_handler(url)
            if inspect.isawaitable(result):
                result = await result
            return result  # type: ignore[no-any-return]

        def stream(self, method: str, url: str, **kwargs: Any) -> _FakeStream:
            assert method == "GET"
            response = get_handler(url) if get_handler else _FakeResponse(200)
            return _FakeStream(response)

    return _Client


async def _allow_all(url: str) -> str | None:
    return None


async def _public_ips(host: str) -> list[str]:
    return ["93.184.216.34"]


def _pin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.citations.link_check.resolve_public_ips", _public_ips)


async def test_check_link_200(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.citations.link_check.validate_url", _allow_all)
    _pin(monkeypatch)
    monkeypatch.setattr(
        "app.citations.link_check.httpx.AsyncClient",
        _fake_client(lambda url: _FakeResponse(200)),
    )
    status = await check_link("https://example.com/a", timeout=1)
    assert status.reachable is True
    assert status.status_code == 200


async def test_check_link_head_403_falls_back_to_get(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.citations.link_check.validate_url", _allow_all)
    _pin(monkeypatch)
    monkeypatch.setattr(
        "app.citations.link_check.httpx.AsyncClient",
        _fake_client(lambda url: _FakeResponse(403), lambda url: _FakeResponse(200)),
    )
    status = await check_link("https://example.com/a", timeout=1)
    assert status.reachable is True
    assert status.status_code == 200


async def test_check_link_404_is_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.citations.link_check.validate_url", _allow_all)
    _pin(monkeypatch)
    monkeypatch.setattr(
        "app.citations.link_check.httpx.AsyncClient",
        _fake_client(lambda url: _FakeResponse(404)),
    )
    status = await check_link("https://example.com/missing", timeout=1)
    assert status.reachable is False
    assert status.status_code == 404


async def test_check_link_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.citations.link_check.validate_url", _allow_all)
    _pin(monkeypatch)

    async def slow(url: str) -> _FakeResponse:
        await asyncio.sleep(0.05)
        return _FakeResponse(200)

    monkeypatch.setattr(
        "app.citations.link_check.httpx.AsyncClient",
        _fake_client(slow),
    )
    status = await check_link("https://example.com/a", timeout=0.01)
    assert status.reachable is False
    assert status.reason == "timeout"


async def test_check_link_redirect_to_private_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def guarded(url: str) -> str | None:
        return "私网" if "127.0.0.1" in url else None

    _pin(monkeypatch)
    calls = {"count": 0}

    def head(url: str) -> _FakeResponse:
        calls["count"] += 1
        return _FakeResponse(301, {"location": "http://127.0.0.1/secret"})

    monkeypatch.setattr("app.citations.link_check.validate_url", guarded)
    monkeypatch.setattr(
        "app.citations.link_check.httpx.AsyncClient",
        _fake_client(head),
    )
    status = await check_link("https://example.com/a", timeout=1)
    assert status.reachable is False
    assert status.rejected is True
    assert calls["count"] == 1


async def test_check_link_redirect_follows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.citations.link_check.validate_url", _allow_all)
    _pin(monkeypatch)
    calls = {"count": 0}

    def head(url: str) -> _FakeResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            return _FakeResponse(302, {"location": "https://target.example.com/final"})
        return _FakeResponse(200)

    monkeypatch.setattr(
        "app.citations.link_check.httpx.AsyncClient",
        _fake_client(head),
    )
    status = await check_link("https://example.com/a", timeout=1)
    assert status.reachable is True
    assert status.status_code == 200
    assert calls["count"] == 2


def test_check_links_limits_and_cache() -> None:
    from app.citations.link_check import LinkCheckCache

    cache = LinkCheckCache(ttl=600, max_size=2)
    cache.put(LinkStatus("https://example.com/a", True, 200))
    assert cache.get("https://example.com/a") is not None
    assert cache.get("https://example.com/a") is not None
    cache.put(LinkStatus("https://example.com/b", True, 200))
    cache.put(LinkStatus("https://example.com/c", True, 200))
    assert cache.get("https://example.com/a") is None


def test_percentile() -> None:
    assert percentile([], 95) is None
    assert percentile([1, 2, 3, 4, 5], 95) == 5
    assert percentile([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 95) == 100


async def test_search_tool_link_failure_does_not_fail_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Provider:
        async def search(
            self,
            query: str,
            *,
            max_results: int = 5,
            recency_days: int | None = None,
            site: str | None = None,
        ) -> list[dict[str, str]]:
            return [
                {
                    "title": "不可达",
                    "url": "https://example.com/timeout",
                    "snippet": "摘要",
                }
            ]

    async def _failed(urls: list[str], **kwargs: Any) -> dict[str, LinkStatus]:
        return {
            url: LinkStatus(url, False, None, "timeout") for url in urls
        }

    monkeypatch.setattr("app.mcp.tools.search.DeepSeekWebSearch", _Provider)
    monkeypatch.setattr("app.mcp.tools.search.check_links", _failed)
    payload = await search_tool("测试", max_results=1)
    assert payload["citations"]
    assert payload["citations"][0]["reachable"] is False
    assert payload["citations"][0]["reason"] == "timeout"
    assert payload["reachable_count"] == 0


def test_csv_formula_injection_and_bom() -> None:
    assert _safe_cell("=1+1") == "'=1+1"
    assert _safe_cell("+cmd") == "'+cmd"
    assert _safe_cell("-2+3") == "'-2+3"
    assert _safe_cell("@sum") == "'@sum"
    assert _safe_cell("normal") == "normal"
    content = _build_csv(
        [
            {
                "started_at": "2026-10-22T10:00:00+00:00",
                "session_id": "s1",
                "turn_no": 1,
                "trace_id": "t1",
                "user_input": "=1+1",
                "final_answer": "ok",
                "model_layer": "deepseek",
                "tools": [{"name": "calculator", "status": "ok", "latency_ms": 5}],
                "duration_ms": 10,
                "tokens": 3,
                "feedback": None,
            }
        ]
    )
    assert content.startswith(b"\xef\xbb\xbf")
    assert "'=1+1" in content.decode("utf-8-sig")
