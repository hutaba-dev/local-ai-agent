"""Compact capability discovery and dynamic tool exposure for AHNBYS agents."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable


class PermissionClass(str, Enum):
    READ = "READ"
    WRITE_SAFE = "WRITE_SAFE"
    WRITE_SENSITIVE = "WRITE_SENSITIVE"
    ADMIN = "ADMIN"


@dataclass(frozen=True)
class CapabilitySpec:
    name: str
    label: str
    description: str
    cost_class: str
    feature_flag: str
    tools: tuple[str, ...]


@dataclass(frozen=True)
class AgentToolSpec:
    name: str
    capability: str
    description: str
    server: str
    cost_class: str
    permission: str
    input_schema: dict[str, object]

    def openai_schema(self) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


CAPABILITIES = (
    CapabilitySpec("web", "Web Search", "Current public web and news discovery.", "very_low_to_paid", "MCP_SEARCH_ENABLED", ("search_web", "search_news", "fetch_page")),
    CapabilitySpec("browser", "Browser", "Read and interact with public JavaScript-rendered pages when fetch is insufficient.", "local_compute", "MCP_PLAYWRIGHT_ENABLED", ("browse_page", "browse_click")),
    CapabilitySpec("time", "Current Time", "Exact current dates, timezones, and time conversion.", "local_compute", "MCP_TIME_ENABLED", ("get_current_time", "convert_time")),
    CapabilitySpec("documentation", "Software Documentation", "Current library and framework documentation with version-specific examples.", "external", "MCP_CONTEXT7_ENABLED", ("resolve_library_id", "query_documentation")),
    CapabilitySpec("github", "GitHub", "Read repository code, history, issues, pull requests, and branches from GitHub.", "external", "MCP_GITHUB_ENABLED", ("github_search_code", "github_read_repository")),
    CapabilitySpec("git", "Local Git", "Read status, diffs, history, commits, and blame in the AHNBYS repository.", "local_compute", "MCP_GIT_ENABLED", ("git_status", "git_log", "git_diff", "git_show", "git_blame")),
    CapabilitySpec("academic", "Academic Research", "Researcher identity, publications, citations, and source provenance.", "varies", "MCP_ACADEMIC_ENABLED", ("researcher_profile", "publication_search")),
    CapabilitySpec("project", "Project Knowledge", "Authorized project files, memories, conversations, and artifacts.", "local_compute", "MCP_PROJECT_ENABLED", ("project_search", "project_list_files", "project_read_file", "project_get_memories")),
    CapabilitySpec("image", "Image Work", "Generate or edit images through the existing AHN7 worker.", "gpu_compute", "MCP_IMAGE_ENABLED", ("generate_image", "edit_image", "adjust_face_pose")),
)


def _object(properties: dict[str, object], required: Iterable[str] = ()) -> dict[str, object]:
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


def _string(description: str, max_length: int = 500) -> dict[str, object]:
    return {"type": "string", "description": description, "maxLength": max_length}


TOOL_SPECS = {
    "search_web": AgentToolSpec("search_web", "web", "Search the current public web. Results are discovery metadata; fetch important pages before relying on them.", "search-mcp", "very_low_to_paid", "READ", _object({"query": _string("Focused search query"), "freshness": {"type": "string", "enum": ["normal", "high"]}}, ("query",))),
    "search_news": AgentToolSpec("search_news", "web", "Search recent public news when freshness materially matters.", "search-mcp", "very_low_to_paid", "READ", _object({"query": _string("Focused news query"), "freshness": {"type": "string", "enum": ["normal", "high"]}}, ("query",))),
    "fetch_page": AgentToolSpec("fetch_page", "web", "Fetch one selected public HTTPS page through the existing secure extraction boundary.", "web-mcp", "very_low", "READ", _object({"url": _string("Public HTTPS URL from a prior result", 2000)}, ("url",))),
    "browse_page": AgentToolSpec("browse_page", "browser", "Open one public HTTPS JavaScript-rendered page through Playwright and return a bounded relevant snapshot.", "browser-mcp", "local_compute", "READ", _object({"url": _string("Public HTTPS URL", 2000), "find_text": _string("Optional text to locate in the rendered page", 300)}, ("url",))),
    "browse_click": AgentToolSpec("browse_click", "browser", "Open a public page, click one exact snapshot reference or unique selector, and return the resulting bounded snapshot. Never bypass access controls or CAPTCHAs.", "browser-mcp", "local_compute", "READ", _object({"url": _string("Public HTTPS URL", 2000), "target": _string("Exact snapshot reference or unique selector", 500), "element": _string("Human-readable element description", 300)}, ("url", "target"))),
    "resolve_library_id": AgentToolSpec("resolve_library_id", "documentation", "Resolve an official library name before querying current Context7 documentation.", "context7-mcp", "external", "READ", _object({"library_name": _string("Official library name", 200), "query": _string("Version-specific documentation need", 1000)}, ("library_name", "query"))),
    "query_documentation": AgentToolSpec("query_documentation", "documentation", "Use this when current library/framework documentation or version-specific implementation details materially improve the answer.", "context7-mcp", "external", "READ", _object({"library_id": _string("Exact Context7 /org/project ID", 300), "query": _string("One specific documentation topic", 1000)}, ("library_id", "query"))),
    "github_search_code": AgentToolSpec("github_search_code", "github", "Search code through the official GitHub MCP server in read-only mode.", "github-mcp", "external", "READ", _object({"query": _string("GitHub code search query", 256)}, ("query",))),
    "github_read_repository": AgentToolSpec("github_read_repository", "github", "Read a GitHub repository file, history, branches, issue, or pull request. No write operations are available.", "github-mcp", "external", "READ", _object({"owner": _string("Repository owner", 100), "repo": _string("Repository name", 100), "operation": {"type": "string", "enum": ["file", "history", "branches", "issue", "pull_request"]}, "path": _string("Optional repository path", 500), "number": {"type": "integer", "minimum": 1}, "ref": _string("Optional branch, tag, or commit reference", 200)}, ("owner", "repo", "operation"))),
    "get_current_time": AgentToolSpec("get_current_time", "time", "Get exact current time in up to five IANA timezones. Use this to resolve relative dates instead of guessing.", "developer-mcp", "local_compute", "READ", _object({"timezones": {"type": "array", "items": _string("IANA timezone", 100), "maxItems": 5}})),
    "convert_time": AgentToolSpec("convert_time", "time", "Convert an ISO date-time between IANA timezones.", "developer-mcp", "local_compute", "READ", _object({"value": _string("ISO 8601 date-time", 100), "from_timezone": _string("Source IANA timezone", 100), "to_timezone": _string("Target IANA timezone", 100)}, ("value", "from_timezone", "to_timezone"))),
    "git_status": AgentToolSpec("git_status", "git", "Read concise local AHNBYS repository status.", "developer-mcp", "local_compute", "READ", _object({})),
    "git_log": AgentToolSpec("git_log", "git", "Read bounded local commit history, optionally for one repository-relative path.", "developer-mcp", "local_compute", "READ", _object({"limit": {"type": "integer", "minimum": 1, "maximum": 50}, "relative_path": _string("Optional repository-relative path")})),
    "git_diff": AgentToolSpec("git_diff", "git", "Read local Git diff without modifying the worktree.", "developer-mcp", "local_compute", "READ", _object({"relative_path": _string("Optional repository-relative path"), "staged": {"type": "boolean"}, "summary": {"type": "boolean"}})),
    "git_show": AgentToolSpec("git_show", "git", "Read one local commit, tag, or branch.", "developer-mcp", "local_compute", "READ", _object({"revision": _string("Revision such as HEAD or a commit hash", 200), "relative_path": _string("Optional repository-relative path")})),
    "git_blame": AgentToolSpec("git_blame", "git", "Read line provenance for one repository-relative file.", "developer-mcp", "local_compute", "READ", _object({"relative_path": _string("Repository-relative file path"), "revision": _string("Revision", 200)}, ("relative_path",))),
    "researcher_profile": AgentToolSpec("researcher_profile", "academic", "Resolve a researcher identity and retrieve a normalized profile with provider-specific provenance.", "academic-mcp", "varies", "READ", _object({"query": _string("Researcher name or identity query")}, ("query",))),
    "publication_search": AgentToolSpec("publication_search", "academic", "Search normalized scholarly publication metadata when academic evidence materially improves the answer.", "academic-mcp", "varies", "READ", _object({"query": _string("Focused publication query"), "limit": {"type": "integer", "minimum": 1, "maximum": 10}}, ("query",))),
    "project_search": AgentToolSpec("project_search", "project", "Search authorized current-project files, memories, and conversations.", "project-mcp", "local_compute", "READ", _object({"query": _string("Current-project search query")}, ("query",))),
    "project_list_files": AgentToolSpec("project_list_files", "project", "List metadata for files in the authorized current project.", "project-mcp", "local_compute", "READ", _object({})),
    "project_read_file": AgentToolSpec("project_read_file", "project", "Read one authorized current-project file by opaque file ID.", "project-mcp", "local_compute", "READ", _object({"file_id": _string("Opaque project file ID", 100)}, ("file_id",))),
    "project_get_memories": AgentToolSpec("project_get_memories", "project", "Read active memories from the authorized current project.", "project-mcp", "local_compute", "READ", _object({"query": _string("Optional memory search query")})),
}


def _enabled(flag: str, default: bool = True) -> bool:
    value = os.getenv(flag)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def capability_catalog(*, project_available: bool = False, image_available: bool = False) -> list[dict[str, object]]:
    catalog = []
    for capability in CAPABILITIES:
        default = False if capability.name == "browser" else True
        enabled = _enabled("MCP_ENABLED", False) and _enabled(capability.feature_flag, default)
        configured = True
        if capability.name == "browser":
            configured = _enabled("MCP_PLAYWRIGHT_EGRESS_GUARD", False)
        elif capability.name == "github":
            configured = bool(os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"))
        elif capability.name == "project":
            configured = project_available
        elif capability.name == "image":
            configured = image_available
        catalog.append({
            "name": capability.name,
            "label": capability.label,
            "description": capability.description,
            "cost_class": capability.cost_class,
            "available": enabled and configured,
            "status": "AVAILABLE" if enabled and configured else ("UNCONFIGURED" if enabled else "DISABLED"),
        })
    return catalog


def detailed_tools(selected_capabilities: Iterable[str], limit: int = 10) -> tuple[AgentToolSpec, ...]:
    selected = set(selected_capabilities)
    ordered = [
        spec for spec in TOOL_SPECS.values()
        if spec.capability in selected
        and (spec.name != "fetch_page" or _enabled("MCP_FETCH_ENABLED", True))
    ]
    return tuple(ordered[: max(1, min(limit, 10))])


def registry_snapshot() -> dict[str, object]:
    return {
        "capabilities": [asdict(capability) for capability in CAPABILITIES],
        "tools": [asdict(tool) for tool in TOOL_SPECS.values()],
    }
