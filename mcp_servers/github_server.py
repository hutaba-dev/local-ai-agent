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
UPSTREAM_TOOLS = (
    "search_code", "get_file_contents", "list_commits", "get_commit",
    "issue_read", "pull_request_read", "search_issues", "list_releases",
    "get_latest_release", "get_release_by_tag",
)

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
    environment.pop("GITHUB_TOOLSETS", None)
    environment.update({
        "GITHUB_PERSONAL_ACCESS_TOKEN": token,
        "GITHUB_READ_ONLY": "1",
    })
    return StdioServerParameters(
        command=str(GITHUB_BINARY),
        args=["stdio", "--read-only", f"--tools={','.join(UPSTREAM_TOOLS)}"],
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
    lowered = text.lower()
    status = "AVAILABLE"
    if result.is_error:
        status = "RATE_LIMITED" if "rate limit" in lowered or "status 429" in lowered else "ERROR"
    elif not text.strip():
        status = "DEGRADED"
    return {
        "status": status,
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


def _repository(owner: str, repo: str) -> dict[str, object]:
    return {"owner": _name(owner, "owner"), "repo": _name(repo, "repository")}


def _number(value: int, label: str) -> int:
    if not 1 <= value <= 100_000_000:
        raise ValueError(f"invalid GitHub {label} number")
    return value


@GITHUB_MCP.tool(description="Read one file or directory from a GitHub repository at an optional ref.", structured_output=True)
async def github_get_file(owner: str, repo: str, path: str = "/", ref: str | None = None) -> dict[str, object]:
    arguments = _repository(owner, repo)
    arguments["path"] = path[:500]
    if ref:
        arguments["ref"] = ref[:200]
    return await _call("get_file_contents", arguments)


@GITHUB_MCP.tool(description="List bounded GitHub commit history or read one commit with file statistics and no full patch.", structured_output=True)
async def github_read_commits(
    owner: str, repo: str, operation: Literal["list", "get"],
    path: str | None = None, ref: str | None = None, sha: str | None = None,
) -> dict[str, object]:
    arguments = _repository(owner, repo)
    if operation == "list":
        arguments.update({"perPage": 20, "fields": ["sha", "html_url", "commit", "author"]})
        if path:
            arguments["path"] = path[:500]
        if ref:
            arguments["sha"] = ref[:200]
        return await _call("list_commits", arguments)
    if not sha:
        raise ValueError("GitHub commit get requires a SHA, branch, or tag")
    arguments.update({"sha": sha[:200], "detail": "stats", "perPage": 50})
    return await _call("get_commit", arguments)


@GITHUB_MCP.tool(description="Read one GitHub issue or search bounded issue and pull-request metadata.", structured_output=True)
async def github_read_issues(
    operation: Literal["get", "search"], owner: str | None = None, repo: str | None = None,
    number: int | None = None, query: str | None = None, include_comments: bool = False,
) -> dict[str, object]:
    if bool(owner) != bool(repo):
        raise ValueError("GitHub owner and repository must be provided together")
    if operation == "search":
        search_query = (query or "").strip()
        if not search_query or len(search_query) > 256:
            raise ValueError("invalid GitHub issue search")
        arguments: dict[str, object] = {"query": search_query, "perPage": 20, "fields": ["number", "title", "state", "html_url", "updated_at", "pull_request"]}
        if owner and repo:
            arguments.update(_repository(owner, repo))
        return await _call("search_issues", arguments)
    if not owner or not repo or number is None:
        raise ValueError("GitHub issue get requires owner, repository, and number")
    arguments = _repository(owner, repo)
    arguments.update({"method": "get_comments" if include_comments else "get", "issue_number": _number(number, "issue"), "perPage": 30})
    return await _call("issue_read", arguments)


@GITHUB_MCP.tool(description="Read one GitHub pull request, its files, commits, reviews, comments, checks, status, or diff.", structured_output=True)
async def github_get_pull_request(
    owner: str, repo: str, number: int,
    view: Literal["details", "files", "commits", "reviews", "review_comments", "comments", "checks", "status", "diff"] = "details",
) -> dict[str, object]:
    methods = {
        "details": "get", "files": "get_files", "commits": "get_commits", "reviews": "get_reviews",
        "review_comments": "get_review_comments", "comments": "get_comments", "checks": "get_check_runs",
        "status": "get_status", "diff": "get_diff",
    }
    arguments = _repository(owner, repo)
    arguments.update({"method": methods[view], "pullNumber": _number(number, "pull request"), "perPage": 30})
    return await _call("pull_request_read", arguments)


@GITHUB_MCP.tool(description="List bounded GitHub releases, read the latest release, or read one release by tag.", structured_output=True)
async def github_read_releases(
    owner: str, repo: str, operation: Literal["list", "latest", "tag"] = "latest", tag: str | None = None,
) -> dict[str, object]:
    arguments = _repository(owner, repo)
    if operation == "list":
        arguments.update({"perPage": 20, "fields": ["tag_name", "name", "html_url", "published_at", "prerelease", "draft"]})
        return await _call("list_releases", arguments)
    if operation == "tag":
        if not tag:
            raise ValueError("GitHub release tag operation requires a tag")
        arguments["tag"] = tag[:200]
        return await _call("get_release_by_tag", arguments)
    return await _call("get_latest_release", arguments)


if __name__ == "__main__":
    GITHUB_MCP.run(transport="stdio")
