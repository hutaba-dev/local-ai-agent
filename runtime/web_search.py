"""Bounded external web search for Research tasks."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass

import httpx

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
NAVER_ENDPOINT = "https://openapi.naver.com/v1/search/webkr.json"
REDDIT_ENDPOINT = "https://www.reddit.com/search.json"
KOREAN_PATTERN = re.compile(r"[\uac00-\ud7a3]")
USER_AGENT = "local-ai-agent-research/0.1"


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