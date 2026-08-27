"""Read-only local time and Git capabilities for AHNBYS."""

from __future__ import annotations

import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from mcp.server import MCPServer


REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_GIT_OUTPUT_CHARS = 12_000
REVISION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@{}^~:-]{0,199}$")

DEVELOPER_MCP = MCPServer(
    "ahnbys-developer",
    description="Read-only current-time and repository-scoped Git capabilities.",
    version="1.0.0",
)


def _timezone(name: str) -> ZoneInfo:
    if len(name) > 100:
        raise ValueError("timezone name is too long")
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as error:
        raise ValueError("unknown IANA timezone") from error


def _repo_path(relative_path: str | None) -> str | None:
    if relative_path is None:
        return None
    candidate = relative_path.strip()
    if not candidate or len(candidate) > 500 or Path(candidate).is_absolute():
        raise ValueError("path must be a non-empty repository-relative path")
    resolved = (REPO_ROOT / candidate).resolve(strict=False)
    if not resolved.is_relative_to(REPO_ROOT.resolve()):
        raise ValueError("path escapes the repository")
    return resolved.relative_to(REPO_ROOT).as_posix()


def _revision(value: str) -> str:
    if not REVISION_PATTERN.fullmatch(value) or value.startswith("-"):
        raise ValueError("invalid Git revision")
    return value


def _git(arguments: list[str]) -> dict[str, object]:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "--no-pager", *arguments],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    output = (result.stdout or result.stderr)[:MAX_GIT_OUTPUT_CHARS]
    return {
        "status": "AVAILABLE" if result.returncode == 0 else "ERROR",
        "repository": REPO_ROOT.name,
        "output": output,
        "truncated": len(result.stdout or result.stderr) > MAX_GIT_OUTPUT_CHARS,
        "exit_code": result.returncode,
    }


@DEVELOPER_MCP.tool(
    description="Get the exact current date and time in one or more IANA timezones. Use this to resolve relative dates instead of guessing.",
    structured_output=True,
)
def get_current_time(timezones: list[str] | None = None) -> dict[str, object]:
    requested = timezones or ["UTC", "Asia/Seoul"]
    if not 1 <= len(requested) <= 5:
        raise ValueError("provide between 1 and 5 timezones")
    now = datetime.now(tz=ZoneInfo("UTC"))
    values = []
    for name in requested:
        zone = _timezone(name)
        local = now.astimezone(zone)
        values.append({
            "timezone": name,
            "iso": local.isoformat(timespec="seconds"),
            "date": local.date().isoformat(),
            "utc_offset": local.strftime("%z"),
            "weekday": local.strftime("%A"),
        })
    return {"status": "AVAILABLE", "times": values}


@DEVELOPER_MCP.tool(
    description="Convert an ISO 8601 date-time between IANA timezones. Use this for UTC, KST, US Eastern, and market-time conversion.",
    structured_output=True,
)
def convert_time(value: str, from_timezone: str, to_timezone: str) -> dict[str, object]:
    source_zone = _timezone(from_timezone)
    target_zone = _timezone(to_timezone)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("value must be an ISO 8601 date-time") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=source_zone)
    converted = parsed.astimezone(target_zone)
    return {
        "status": "AVAILABLE",
        "from": parsed.isoformat(timespec="seconds"),
        "to": converted.isoformat(timespec="seconds"),
        "timezone": to_timezone,
        "weekday": converted.strftime("%A"),
    }


@DEVELOPER_MCP.tool(description="Inspect concise local repository status without modifying the worktree.", structured_output=True)
def git_status() -> dict[str, object]:
    return _git(["status", "--short", "--branch"])


@DEVELOPER_MCP.tool(description="Read bounded local commit history for the AHNBYS repository.", structured_output=True)
def git_log(limit: int = 10, relative_path: str | None = None) -> dict[str, object]:
    if not 1 <= limit <= 50:
        raise ValueError("limit must be between 1 and 50")
    arguments = ["log", f"-{limit}", "--date=iso-strict", "--pretty=format:%h%x09%ad%x09%an%x09%s"]
    path = _repo_path(relative_path)
    if path:
        arguments.extend(["--", path])
    return _git(arguments)


@DEVELOPER_MCP.tool(description="Read the current local Git diff or diff summary for one repository-relative path.", structured_output=True)
def git_diff(relative_path: str | None = None, staged: bool = False, summary: bool = False) -> dict[str, object]:
    arguments = ["diff"]
    if staged:
        arguments.append("--cached")
    if summary:
        arguments.append("--stat")
    path = _repo_path(relative_path)
    if path:
        arguments.extend(["--", path])
    return _git(arguments)


@DEVELOPER_MCP.tool(description="Read one local commit, tag, or branch without changing repository state.", structured_output=True)
def git_show(revision: str = "HEAD", relative_path: str | None = None) -> dict[str, object]:
    arguments = ["show", "--stat", "--oneline", _revision(revision)]
    path = _repo_path(relative_path)
    if path:
        arguments.extend(["--", path])
    return _git(arguments)


@DEVELOPER_MCP.tool(description="Read line provenance for one repository-relative text file.", structured_output=True)
def git_blame(relative_path: str, revision: str = "HEAD") -> dict[str, object]:
    return _git(["blame", "--line-porcelain", _revision(revision), "--", _repo_path(relative_path) or ""])


if __name__ == "__main__":
    DEVELOPER_MCP.run(transport="stdio")
