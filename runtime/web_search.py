"""Bounded external web search for Research tasks."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import httpx


BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    description: str


def search(query: str, mode: str) -> list[dict[str, str]]:
    api_key = os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        raise RuntimeError("web search is required but BRAVE_SEARCH_API_KEY is not configured")
    result_count = 5 if mode == "QUICK_SEARCH" else 8
    response = httpx.get(
        BRAVE_ENDPOINT,
        headers={"Accept": "application/json", "X-Subscription-Token": api_key},
        params={"q": query, "count": result_count, "safesearch": "moderate"},
        timeout=12,
    )
    response.raise_for_status()
    items = response.json().get("web", {}).get("results", [])
    return [
        asdict(SearchResult(title=item["title"], url=item["url"], description=item.get("description", "")))
        for item in items[:result_count]
        if isinstance(item, dict) and isinstance(item.get("title"), str) and isinstance(item.get("url"), str)
    ]