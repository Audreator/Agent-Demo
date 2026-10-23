from __future__ import annotations

import asyncio
import ipaddress
import os
import socket
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

MAX_URL_LENGTH = 2048
MAX_REDIRECTS = 5
REDIRECT_CODES = {301, 302, 303, 307, 308}
FALLBACK_CODES = {400, 403, 405, 429}
MAX_PORT = 65535
DEFAULT_TIMEOUT = 8.0
DEFAULT_CONCURRENCY = 5
DEFAULT_CACHE_TTL = 600
DEFAULT_CACHE_SIZE = 500
USER_AGENT = "minimal-agent-link-check/1.0"

BLOCKED_NETWORKS = (
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
)


@dataclass
class LinkStatus:
    url: str
    reachable: bool
    status_code: int | None = None
    reason: str = ""
    rejected: bool = False
    checked_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def as_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "reachable": self.reachable,
            "status_code": self.status_code,
            "reason": self.reason,
            "rejected": self.rejected,
            "checked_at": self.checked_at.isoformat(),
        }


def normalize_url(url: str) -> str:
    """协议/域名小写、去 fragment、保留 query、限制长度。"""
    raw = str(url or "").strip()
    if not raw or len(raw) > MAX_URL_LENGTH:
        return ""
    try:
        parts = urlsplit(raw)
        scheme = parts.scheme.lower()
        hostname = parts.hostname
    except (ValueError, UnicodeError):
        return ""
    if scheme not in {"http", "https"} or not hostname:
        return ""
    if parts.username or parts.password:
        return ""
    try:
        return urlunsplit((scheme, parts.netloc.lower(), parts.path, parts.query, ""))
    except (ValueError, UnicodeError):
        return ""


def _is_public_ip(value: str) -> bool:
    try:
        ip = ipaddress.ip_address(value.split("%", 1)[0])
    except ValueError:
        return False
    if ip.version == 6 and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if any(ip in network for network in BLOCKED_NETWORKS):
        return False
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


async def resolve_public_ips(host: str) -> list[str] | None:
    """解析域名并返回全部公网 IP；任一解析结果为私网则拒绝。"""
    if host.lower() in {"localhost", "localhost.localdomain"} or host.lower().endswith(
        ".localhost"
    ):
        return None
    try:
        infos = await asyncio.to_thread(
            socket.getaddrinfo, host, None, type=socket.SOCK_STREAM
        )
    except (socket.gaierror, OSError):
        return None
    if not infos:
        return None
    ips: list[str] = []
    for info in infos:
        address = str(info[4][0])
        if not _is_public_ip(address):
            return None
        if address not in ips:
            ips.append(address)
    return ips


async def _resolve_host(host: str) -> bool:
    return await resolve_public_ips(host) is not None


async def validate_url(url: str) -> str | None:
    """返回拒绝原因；None 表示允许发起请求。"""
    normalized = normalize_url(url)
    if not normalized:
        return "仅允许 http/https 且长度不超过 2048"
    parts = urlsplit(normalized)
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        return "端口非法"
    if not 1 <= port <= MAX_PORT:
        return "端口非法"
    if not await _resolve_host(parts.hostname or ""):
        return "域名解析到私网或不可解析"
    return None


def _pin_url(url: str, ip: str) -> tuple[str, dict[str, str], dict[str, Any]]:
    """把请求固定到已校验的 IP，避免 DNS rebinding；Host/SNI 保留原域名。"""
    parts = urlsplit(url)
    host = parts.hostname or ""
    ip_host = f"[{ip}]" if ":" in ip else ip
    netloc = ip_host + (f":{parts.port}" if parts.port else "")
    pinned = urlunsplit((parts.scheme, netloc, parts.path, parts.query, ""))
    headers = {"Host": parts.netloc}
    extensions: dict[str, Any] = {}
    if parts.scheme == "https":
        extensions["sni_hostname"] = host
    return pinned, headers, extensions


@dataclass
class _Probe:
    status_code: int
    location: str | None = None


async def _get_probe(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    extensions: dict[str, Any],
) -> _Probe:
    async with client.stream(
        "GET",
        url,
        headers={**headers, "Range": "bytes=0-2048"},
        follow_redirects=False,
        extensions=extensions,
    ) as response:
        if response.status_code < 400:
            async for _ in response.aiter_bytes(2048):
                break
        return _Probe(response.status_code, response.headers.get("location"))


async def check_link(url: str, timeout: float = DEFAULT_TIMEOUT) -> LinkStatus:
    """HEAD 优先、必要时 GET 探测；手动跟随重定向，固定公网 IP 连接。"""
    original = normalize_url(url)
    if not original:
        return LinkStatus(str(url), False, None, "仅允许 http/https 且长度不超过 2048", True)

    async def _run() -> LinkStatus:
        current = original
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            for _ in range(MAX_REDIRECTS + 1):
                problem = await validate_url(current)
                if problem:
                    return LinkStatus(original, False, None, problem, True)
                parts = urlsplit(current)
                ips = await resolve_public_ips(parts.hostname or "")
                if not ips:
                    return LinkStatus(
                        original, False, None, "域名解析到私网或不可解析", True
                    )
                pinned, headers, extensions = _pin_url(current, ips[0])
                probe: _Probe
                try:
                    head = await client.head(
                        pinned,
                        headers=headers,
                        follow_redirects=False,
                        extensions=extensions,
                    )
                    probe = _Probe(head.status_code, head.headers.get("location"))
                except httpx.HTTPError:
                    try:
                        probe = await _get_probe(client, pinned, headers, extensions)
                    except httpx.HTTPError as exc:
                        return LinkStatus(original, False, None, f"请求失败: {exc}")
                if probe.status_code in FALLBACK_CODES:
                    try:
                        probe = await _get_probe(client, pinned, headers, extensions)
                    except httpx.HTTPError as exc:
                        return LinkStatus(original, False, probe.status_code, f"请求失败: {exc}")
                if probe.status_code in REDIRECT_CODES:
                    if not probe.location:
                        return LinkStatus(
                            original, True, probe.status_code, "重定向缺少 Location"
                        )
                    current = urljoin(current, probe.location)
                    continue
                reachable = 200 <= probe.status_code < 400
                reason = "" if reachable else f"HTTP {probe.status_code}"
                return LinkStatus(original, reachable, probe.status_code, reason)
        return LinkStatus(original, False, None, "重定向次数过多")

    try:
        return await asyncio.wait_for(_run(), timeout=timeout)
    except TimeoutError:
        return LinkStatus(original, False, None, "timeout")
    except Exception as exc:  # 校验失败不影响 search 成功
        return LinkStatus(original, False, None, f"{type(exc).__name__}: {exc}")


class LinkCheckCache:
    """进程内 LRU：默认 500 条、10 分钟。"""

    def __init__(self, ttl: int = DEFAULT_CACHE_TTL, max_size: int = DEFAULT_CACHE_SIZE) -> None:
        self.ttl = max(0, int(ttl))
        self.max_size = max(1, int(max_size))
        self._items: OrderedDict[str, tuple[float, LinkStatus]] = OrderedDict()

    def get(self, url: str) -> LinkStatus | None:
        key = normalize_url(url)
        if not key:
            return None
        item = self._items.get(key)
        if item is None:
            return None
        stored_at, status = item
        if self.ttl and (datetime.now(UTC).timestamp() - stored_at) > self.ttl:
            self._items.pop(key, None)
            return None
        self._items.move_to_end(key)
        return status

    def put(self, status: LinkStatus) -> None:
        key = normalize_url(status.url)
        if not key:
            return
        self._items[key] = (datetime.now(UTC).timestamp(), status)
        self._items.move_to_end(key)
        while len(self._items) > self.max_size:
            self._items.popitem(last=False)


_default_cache = LinkCheckCache(
    ttl=int(os.environ.get("LINK_CHECK_CACHE_TTL", DEFAULT_CACHE_TTL))
)


async def check_links(
    urls: list[str],
    *,
    timeout: float | None = None,
    concurrency: int | None = None,
    cache: LinkCheckCache | None = None,
) -> dict[str, LinkStatus]:
    timeout = float(os.environ.get("LINK_CHECK_TIMEOUT", timeout or DEFAULT_TIMEOUT))
    concurrency = max(
        1, int(os.environ.get("LINK_CHECK_CONCURRENCY", concurrency or DEFAULT_CONCURRENCY))
    )
    store = cache if cache is not None else _default_cache
    semaphore = asyncio.Semaphore(concurrency)
    unique: list[str] = []
    seen: set[str] = set()
    for url in urls:
        key = normalize_url(url)
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(key)

    async def _one(url: str) -> LinkStatus:
        cached = store.get(url)
        if cached is not None:
            return cached
        async with semaphore:
            status = await check_link(url, timeout=timeout)
        store.put(status)
        return status

    checked = await asyncio.gather(*(_one(url) for url in unique), return_exceptions=True)
    result: dict[str, LinkStatus] = {}
    for url, item in zip(unique, checked, strict=False):
        if isinstance(item, BaseException):
            result[url] = LinkStatus(url, False, None, f"{type(item).__name__}: {item}")
        else:
            result[url] = item
    return result


__all__ = [
    "LinkCheckCache",
    "LinkStatus",
    "check_link",
    "check_links",
    "normalize_url",
    "validate_url",
]
