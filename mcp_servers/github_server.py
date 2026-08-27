"""Minimal read-only facade for the official GitHub MCP server."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

from mcp import Client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.server import MCPServer


REPO_ROOT = Path(__file__).resolve().parents[1]
GITHUB_BINARY = REPO_ROOT / "mcp_external" / "bin" / "github-mcp-server"
MAX_GITHUB_OUTPUT_CHARS = 12_000
NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")

GITHUB_MCP = MCPServer(
    "ahnbys-github",
    description="Minimal read-only facade for the official GitHub MCP server.",
    version="1.0.0",
)


def _parameters() -> StdioServerParameters:
    token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("GitHub MCP is unconfigured")
    if not GITHUB_BINARY.is_file():
        raise RuntimeError("GitHub MCP binary is not installed")
    environment = dict(os.environ)
    environment.update({
        "GITHUB_PERSONAL_ACCESS_TOKEN": token,
        "GITHUB_READ_ONLY": "1",
        "GITHUB_TOOLSETS": "repos,issues,pull_requests",
    })
    return StdioServerParameters(
        command=str(GITHUB_BINARY),
        args=["stdio", "--read-only", "--toolsets=repos,issues,pull_requests"],
        env=environment,
        cwd=str(REPO_ROOT),
    )


def _name(value: str, label: str) -> str:
    if not NAME_PATTERN.fullmatch(value):
        raise ValueError(f"invalid GitHub {label}")
    return value


def _text(result: Any) -> str:
    values = []
    for item in getattr(result, "content", ()):
        text = getattr(item, "text", None)
        if isinstance(text, str):
            values.append(text)
    return "\n".join(values)[:MAX_GITHUB_OUTPUT_CHARS]


async def _call(tool: str, arguments: dict[str, object]) -> dict[str, object]:
    async with Client(stdio_client(_parameters()), read_timeout_seconds=30) as client:
        result = await client.call_tool(tool, arguments, read_timeout_seconds=30)
    text = _text(result)
    return {
        "status": "ERROR" if result.is_error else "AVAILABLE",
        "source": "GitHub MCP",
        "output": text,
        "truncated": len(text) >= MAX_GITHUB_OUTPUT_CHARS,
    }


@GITHUB_MCP.tool(description="Search code through the official GitHub MCP server in read-only mode.", structured_output=True)
async def github_search_code(query: str) -> dict[str, object]:
    query = query.strip()
    if not query or len(query) > 256:
        raise ValueError("GitHub code query must contain between 1 and 256 characters")
    return await _call("search_code", {
        "query": query,
        "perPage": 20,
        "fields": ["name", "path", "sha", "repository", "text_matches"],
    })


@GITHUB_MCP.tool(
    description="Read a repository file, commit history, branches, issue, or pull request through the official GitHub MCP server.",
    structured_output=True,
)
async def github_read_repository(
    owner: str,
    repo: str,
    operation: Literal["file", "history", "branches", "issue", "pull_request"],
    path: str | None = None,
    number: int | None = None,
    ref: str | None = None,
) -> dict[str, object]:
    arguments: dict[str, object] = {"owner": _name(owner, "owner"), "repo": _name(repo, "repository")}
    if operation == "file":
        arguments["path"] = (path or "/")[:500]
        if ref:
            arguments["ref"] = ref[:200]
        return await _call("get_file_contents", arguments)
    if operation == "history":
        arguments.update({"perPage": 20, "fields": ["sha", "html_url", "commit", "author"]})
        if path:
            arguments["path"] = path[:500]
        return await _call("list_commits", arguments)
    if operation == "branches":
        arguments["perPage"] = 50
        return await _call("list_branches", arguments)
    if number is None or not 1 <= number <= 100_000_000:
        raise ValueError("issue and pull request operations require a positive number")
    if operation == "issue":
        arguments.update({"method": "get", "issue_number": number})
        return await _call("issue_read", arguments)
    arguments.update({"method": "get", "pullNumber": number})
    return await _call("pull_request_read", arguments)


if __name__ == "__main__":
    GITHUB_MCP.run(transport="stdio")
