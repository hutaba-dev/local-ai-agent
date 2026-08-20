"""Bounded external web search for Research tasks."""

from __future__ import annotations

import os
import re
import ipaddress
import socket
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import httpx

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
NAVER_ENDPOINT = "https://openapi.naver.com/v1/search/webkr.json"
REDDIT_ENDPOINT = "https://www.reddit.com/search.json"
KOREAN_PATTERN = re.compile(r"[\uac00-\ud7a3]")
USER_AGENT = "local-ai-agent-research/0.1"
MAX_SOURCE_COUNT = 5
MAX_SOURCE_CHARS = 6_000


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self._parts.append(data)

    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self._parts)).strip()


@dataclass(frozen=True)
class SearchResult:
    provider: str
    title: str
    url: str
    description: str


def search(query: str, mode: str) -> list[dict[str, str]]:
    """Return bounded search results chosen for Korean or global intent.

    Deep research may include Reddit discussion results as non-authoritative
    context. Provider failures are recorded only when no primary provider works.
    """
    result_count = 5 if mode == "QUICK_SEARCH" else 8
    providers = _primary_providers(query)
    results: list[SearchResult] = []
    errors: list[str] = []
    for provider in providers:
        try:
            results.extend(provider(query, result_count))
        except (RuntimeError, httpx.HTTPError) as error:
            errors.append(str(error))
    if mode == "DEEP_RESEARCH":
        try:
            results.extend(_reddit_search(query, min(3, result_count)))
        except httpx.HTTPError:
            pass
    if not results:
        detail = "; ".join(errors) or "no configured search provider returned results"
        raise RuntimeError(f"web search is required but unavailable: {detail}")
    return [asdict(result) for result in _unique(results)[:result_count]]


def fetch_sources(results: list[dict[str, str]], limit: int = MAX_SOURCE_COUNT) -> list[dict[str, str]]:
    """Fetch bounded text from public HTTPS result URLs for Deep Research.

    URLs are provider output but remain untrusted. Block private network targets,
    validate each redirect, and accept HTML only.
    """
    sources: list[dict[str, str]] = []
    for result in results:
        if len(sources) >= limit:
            break
        url = result.get("url")
        if not isinstance(url, str):
            continue
        try:
            response, final_url = _safe_fetch(url)
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" not in content_type:
                continue
            extractor = _TextExtractor()
            extractor.feed(response.text[:MAX_SOURCE_CHARS * 3])
            text = extractor.text()[:MAX_SOURCE_CHARS]
            if text:
                sources.append({
                    "title": result.get("title", "Untitled source"),
                    "url": final_url,
                    "text": text,
                })
        except (OSError, ValueError, httpx.HTTPError):
            continue
    return sources


def _safe_fetch(url: str) -> tuple[httpx.Response, str]:
    current_url = url
    for _ in range(4):
        _validate_public_https_url(current_url)
        response = httpx.get(
            current_url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
            timeout=12,
            follow_redirects=False,
        )
        if response.is_redirect:
            location = response.headers.get("location")
            if not location:
                raise ValueError("redirect had no location")
            current_url = urljoin(current_url, location)
            continue
        response.raise_for_status()
        return response, current_url
    raise ValueError("too many redirects")


def _validate_public_https_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("only public HTTPS URLs are allowed")
    for address_info in socket.getaddrinfo(parsed.hostname, None, type=socket.SOCK_STREAM):
        address = ipaddress.ip_address(address_info[4][0])
        if not address.is_global:
            raise ValueError("non-public network targets are not allowed")


def _primary_providers(query: str):
    providers = []
    if KOREAN_PATTERN.search(query) and _naver_configured():
        providers.append(_naver_search)
    if os.getenv("BRAVE_SEARCH_API_KEY"):
        providers.append(_brave_search)
    if not providers and _naver_configured():
        providers.append(_naver_search)
    if not providers:
        raise RuntimeError("configure BRAVE_SEARCH_API_KEY or NAVER_SEARCH_CLIENT_ID and NAVER_SEARCH_CLIENT_SECRET")
    return providers


def _naver_configured() -> bool:
    return bool(os.getenv("NAVER_SEARCH_CLIENT_ID") and os.getenv("NAVER_SEARCH_CLIENT_SECRET"))


def _naver_search(query: str, result_count: int) -> list[SearchResult]:
    response = httpx.get(
        NAVER_ENDPOINT,
        headers={
            "X-Naver-Client-Id": os.environ["NAVER_SEARCH_CLIENT_ID"],
            "X-Naver-Client-Secret": os.environ["NAVER_SEARCH_CLIENT_SECRET"],
        },
        params={"query": query, "display": result_count, "sort": "sim"},
        timeout=12,
    )
    response.raise_for_status()
    return [
        SearchResult("naver", _strip_html(item.get("title", "")), item["link"], _strip_html(item.get("description", "")))
        for item in response.json().get("items", [])[:result_count]
        if isinstance(item, dict) and isinstance(item.get("link"), str)
    ]


def _brave_search(query: str, result_count: int) -> list[SearchResult]:
    response = httpx.get(
        BRAVE_ENDPOINT,
        headers={"Accept": "application/json", "X-Subscription-Token": os.environ["BRAVE_SEARCH_API_KEY"]},
        params={"q": query, "count": result_count, "safesearch": "moderate"},
        timeout=12,
    )
    response.raise_for_status()
    return [
        SearchResult("brave", item["title"], item["url"], item.get("description", ""))
        for item in response.json().get("web", {}).get("results", [])[:result_count]
        if isinstance(item, dict) and isinstance(item.get("title"), str) and isinstance(item.get("url"), str)
    ]


def _reddit_search(query: str, result_count: int) -> list[SearchResult]:
    response = httpx.get(
        REDDIT_ENDPOINT,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        params={"q": query, "limit": result_count, "sort": "relevance", "t": "year", "raw_json": 1},
        timeout=12,
    )
    response.raise_for_status()
    children = response.json().get("data", {}).get("children", [])
    return [
        SearchResult("reddit", item["data"]["title"], f"https://www.reddit.com{item['data']['permalink']}", item["data"].get("selftext", "")[:400])
        for item in children[:result_count]
        if isinstance(item, dict) and isinstance(item.get("data"), dict) and isinstance(item["data"].get("title"), str) and isinstance(item["data"].get("permalink"), str)
    ]


def _unique(results: list[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    unique_results = []
    for result in results:
        if result.url not in seen:
            seen.add(result.url)
            unique_results.append(result)
    return unique_results


def _strip_html(value: object) -> str:
    return re.sub(r"<[^>]+>", "", value) if isinstance(value, str) else ""