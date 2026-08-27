"""MCP search tools backed by the existing conditional SearchRouter."""

from __future__ import annotations

from typing import Literal

from mcp.server import MCPServer

from runtime.search_providers import ProviderStatus, SearchRequest, SearchRouter


SEARCH_WEB_DESCRIPTION = (
    "Search the current public web. Use this when up-to-date or external information is likely needed. "
    "Search results are discovery metadata; use fetch_page for important sources before treating them as evidence."
)
SEARCH_NEWS_DESCRIPTION = (
    "Search recent news. Prefer this for current events, company developments, markets, announcements, and breaking developments. "
    "Use fetch_page on important results before treating snippets as evidence."
)

search_mcp = MCPServer(
    "ahnbys-search",
    description="High-level public web discovery backed by the existing AHNBYS Search Router.",
    version="1.0.0",
)


def _search(
    query: str,
    category: Literal["web", "news"],
    max_results: int,
    freshness: str | None,
    provider_hint: Literal["auto", "searxng", "serper", "brave"],
) -> dict[str, object]:
    query = query.strip()
    if not query or len(query) > 500:
        raise ValueError("query must contain between 1 and 500 characters")
    if not 1 <= max_results <= 10:
        raise ValueError("max_results must be between 1 and 10")
    if freshness not in {None, "day", "week", "month", "high", "normal"}:
        raise ValueError("unsupported freshness value")
    normalized_freshness = "VERY_HIGH" if freshness in {"day", "high"} else freshness
    request = SearchRequest(
        query=query,
        category=category,
        freshness=normalized_freshness,
        count=max_results,
        language="ko" if any("가" <= character <= "힣" for character in query) else "en",
    )
    router = SearchRouter()
    if provider_hint == "auto":
        batch = router.search(request, {"freshness": normalized_freshness})
        return {
            "status": "AVAILABLE",
            "query": query,
            "category": category,
            "results": list(batch.results)[:max_results],
            "metrics": batch.metrics,
        }
    provider = router.providers.get(provider_hint)
    if provider is None or not provider.configured():
        return {
            "status": "UNCONFIGURED",
            "query": query,
            "category": category,
            "results": [],
            "metrics": {"provider": provider_hint},
        }
    response = provider.search(request)
    return {
        "status": response.status.value,
        "query": query,
        "category": category,
        "results": [result.to_dict() for result in response.results[:max_results]],
        "metrics": {
            "provider": provider_hint,
            "latency_ms": response.latency_ms,
            "cache_hit": response.cache_hit,
            "http_status": response.http_status,
            "error": response.error,
            "rate_limited": response.status == ProviderStatus.RATE_LIMITED,
        },
    }


@search_mcp.tool(description=SEARCH_WEB_DESCRIPTION, structured_output=True)
def search_web(
    query: str,
    max_results: int = 8,
    freshness: str | None = None,
    provider_hint: Literal["auto", "searxng", "serper", "brave"] = "auto",
) -> dict[str, object]:
    return _search(query, "web", max_results, freshness, provider_hint)


@search_mcp.tool(description=SEARCH_NEWS_DESCRIPTION, structured_output=True)
def search_news(
    query: str,
    max_results: int = 8,
    freshness: str | None = "day",
    provider_hint: Literal["auto", "searxng", "serper", "brave"] = "auto",
) -> dict[str, object]:
    return _search(query, "news", max_results, freshness, provider_hint)


if __name__ == "__main__":
    search_mcp.run(transport="stdio")
