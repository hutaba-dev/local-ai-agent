"""Restricted facade for the official Microsoft Playwright MCP server."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from mcp import Client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.server import MCPServer

from runtime.web_search import _safe_fetch, _validate_public_https_url


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAYWRIGHT_COMMAND = REPO_ROOT / "mcp_external" / "node_modules" / ".bin" / "playwright-mcp"
PLAYWRIGHT_CONFIG = REPO_ROOT / "mcp_external" / "playwright.config.json"
MAX_BROWSER_OUTPUT_CHARS = 12_000

BROWSER_MCP = MCPServer(
    "ahnbys-browser",
    description="Restricted public-web browser facade backed by the official Microsoft Playwright MCP server.",
    version="1.0.0",
)


def _parameters(allowed_origin: str) -> StdioServerParameters:
    if not PLAYWRIGHT_COMMAND.is_file():
        raise RuntimeError("Playwright MCP is not installed")
    return StdioServerParameters(
        command=str(PLAYWRIGHT_COMMAND),
        args=["--config", str(PLAYWRIGHT_CONFIG), "--allowed-origins", allowed_origin],
        env=dict(os.environ),
        cwd=str(REPO_ROOT),
    )


def _text(result: Any) -> str:
    values = []
    for item in getattr(result, "content", ()):
        text = getattr(item, "text", None)
        if isinstance(text, str):
            values.append(text)
    return "\n".join(values)[:MAX_BROWSER_OUTPUT_CHARS]


def _prepare_url(url: str) -> tuple[str, str]:
    if os.getenv("MCP_PLAYWRIGHT_EGRESS_GUARD", "").strip().lower() not in {"1", "true", "yes", "on"}:
        raise RuntimeError("Playwright MCP requires a verified network egress guard")
    _validate_public_https_url(url)
    _, final_url = _safe_fetch(url)
    _validate_public_https_url(final_url)
    parsed = urlparse(final_url)
    return final_url, f"{parsed.scheme}://{parsed.netloc}"


async def _navigate(client: Client, url: str) -> Any:
    return await client.call_tool("browser_navigate", {"url": url}, read_timeout_seconds=30)


@BROWSER_MCP.tool(
    description="Open one public HTTPS JavaScript-rendered page and return a bounded relevant snapshot. Use when secure fetch is insufficient.",
    structured_output=True,
)
async def browse_page(url: str, find_text: str | None = None) -> dict[str, object]:
    if len(url) > 2_000:
        raise ValueError("url is too long")
    if find_text is not None and (not find_text.strip() or len(find_text) > 300):
        raise ValueError("find_text must contain between 1 and 300 characters")
    final_url, allowed_origin = _prepare_url(url)
    async with Client(stdio_client(_parameters(allowed_origin)), read_timeout_seconds=35) as client:
        result = await _navigate(client, final_url)
        if not result.is_error and find_text:
            result = await client.call_tool("browser_find", {"text": find_text.strip()}, read_timeout_seconds=10)
    text = _text(result)
    return {
        "status": "ERROR" if result.is_error else "AVAILABLE",
        "url": final_url,
        "relevant_text": text,
        "truncated": len(text) >= MAX_BROWSER_OUTPUT_CHARS,
        "engine": "Playwright MCP",
    }


@BROWSER_MCP.tool(
    description="Open a public HTTPS page, click one exact element reference or unique selector, and return the resulting bounded snapshot.",
    structured_output=True,
)
async def browse_click(url: str, target: str, element: str = "selected public page element") -> dict[str, object]:
    if len(url) > 2_000 or not target.strip() or len(target) > 500 or len(element) > 300:
        raise ValueError("invalid browser interaction arguments")
    final_url, allowed_origin = _prepare_url(url)
    async with Client(stdio_client(_parameters(allowed_origin)), read_timeout_seconds=35) as client:
        navigated = await _navigate(client, final_url)
        if navigated.is_error:
            result = navigated
        else:
            result = await client.call_tool("browser_click", {
                "target": target.strip(),
                "element": element.strip(),
            }, read_timeout_seconds=20)
    text = _text(result)
    return {
        "status": "ERROR" if result.is_error else "AVAILABLE",
        "url": final_url,
        "relevant_text": text,
        "truncated": len(text) >= MAX_BROWSER_OUTPUT_CHARS,
        "engine": "Playwright MCP",
    }


if __name__ == "__main__":
    BROWSER_MCP.run(transport="stdio")
