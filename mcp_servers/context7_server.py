"""AHNBYS facade for the official Context7 MCP service."""

from __future__ import annotations

import os
import re
from typing import Any

from mcp import Client
from mcp.server import MCPServer


CONTEXT7_URL = os.getenv("CONTEXT7_MCP_URL", "https://mcp.context7.com/mcp")
MAX_CONTEXT7_OUTPUT_CHARS = 12_000
SENSITIVE_PATTERN = re.compile(
    r"(?i)(?:api[_ -]?key|password|passwd|secret|access[_ -]?token|bearer)\s*[:=]\s*\S+"
)
RATE_LIMIT_PATTERN = re.compile(r"(?i)rate.?limit|too many requests|\b429\b")

CONTEXT7_MCP = MCPServer(
    "ahnbys-context7",
    description="Bounded facade for current public software documentation from the official Context7 MCP service.",
    version="1.0.0",
)


def _validate_query(value: str) -> str:
    value = value.strip()
    if not value or len(value) > 1_000:
        raise ValueError("documentation query must contain between 1 and 1000 characters")
    if SENSITIVE_PATTERN.search(value):
        raise ValueError("documentation queries must not contain credentials or secrets")
    return value


def _text(result: Any) -> str:
    values = []
    for item in getattr(result, "content", ()):
        text = getattr(item, "text", None)
        if isinstance(text, str):
            values.append(text)
    return "\n".join(values)[:MAX_CONTEXT7_OUTPUT_CHARS]


def _upstream_status(is_error: bool, text: str) -> str:
    if is_error:
        return "RATE_LIMITED" if RATE_LIMIT_PATTERN.search(text) else "ERROR"
    return "AVAILABLE" if text.strip() else "DEGRADED"


async def _call(tool: str, arguments: dict[str, object]) -> dict[str, object]:
    headers = {}
    api_key = os.getenv("CONTEXT7_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    target: str | object = CONTEXT7_URL
    if headers:
        from mcp.client.streamable_http import StreamableHTTPTransport

        target = StreamableHTTPTransport(CONTEXT7_URL, headers=headers)
    async with Client(target, read_timeout_seconds=20) as client:
        result = await client.call_tool(tool, arguments, read_timeout_seconds=20)
    text = _text(result)
    return {
        "status": _upstream_status(result.is_error, text),
        "source": "Context7",
        "text": text,
        "truncated": len(text) >= MAX_CONTEXT7_OUTPUT_CHARS,
    }


@CONTEXT7_MCP.tool(
    description="Resolve an official library name to a Context7 library ID. Use this before documentation lookup unless the user already supplied an exact /org/project ID.",
    structured_output=True,
)
async def resolve_library_id(library_name: str, query: str) -> dict[str, object]:
    library_name = library_name.strip()
    if not library_name or len(library_name) > 200:
        raise ValueError("library_name must contain between 1 and 200 characters")
    return await _call("resolve-library-id", {
        "libraryName": library_name,
        "query": _validate_query(query),
    })


@CONTEXT7_MCP.tool(
    description="Use this when current library/framework documentation or version-specific implementation details materially improve the answer.",
    structured_output=True,
)
async def query_documentation(library_id: str, query: str) -> dict[str, object]:
    library_id = library_id.strip()
    if not re.fullmatch(r"/[A-Za-z0-9_.-]+/[A-Za-z0-9_.@/-]+", library_id) or len(library_id) > 300:
        raise ValueError("library_id must be an exact Context7 /org/project or /org/project/version ID")
    return await _call("query-docs", {
        "libraryId": library_id,
        "query": _validate_query(query),
    })


if __name__ == "__main__":
    CONTEXT7_MCP.run(transport="stdio")
