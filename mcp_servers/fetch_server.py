"""MCP page fetch tool backed by the existing secure source extractor."""

from __future__ import annotations

from typing import Literal
from urllib.parse import urlparse

from mcp.server import MCPServer

from runtime.web_search import fetch_sources


FETCH_PAGE_DESCRIPTION = (
    "Fetch and extract the content of a selected public webpage. Use this after search when the page may contain evidence required "
    "to answer the user's question. Public HTTPS, redirect, content type, timeout, and size restrictions are enforced."
)

fetch_mcp = MCPServer(
    "ahnbys-fetch",
    description="Secure public webpage extraction backed by the existing AHNBYS fetch boundary.",
    version="1.0.0",
)


@fetch_mcp.tool(description=FETCH_PAGE_DESCRIPTION, structured_output=True)
def fetch_page(
    url: str,
    extract_mode: Literal["article"] = "article",
) -> dict[str, object]:
    if len(url) > 2_000:
        raise ValueError("url is too long")
    sources, fetches = fetch_sources(
        [{"title": urlparse(url).hostname or "Selected source", "url": url}],
        limit=1,
        include_metrics=True,
    )
    measurement = fetches[0] if fetches else {"success": False, "failure_reason": "fetch_not_attempted"}
    if not sources:
        return {
            "status": "ERROR",
            "url": url,
            "title": "",
            "text": "",
            "published_at": None,
            "metadata": measurement,
        }
    source = sources[0]
    return {
        "status": "AVAILABLE",
        "url": source["url"],
        "title": source["title"],
        "text": source["text"],
        "published_at": None,
        "metadata": {"extract_mode": extract_mode, **measurement},
    }


if __name__ == "__main__":
    fetch_mcp.run(transport="stdio")
