"""Compact capability discovery and dynamic tool exposure for AHNBYS agents."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Iterable


class PermissionClass(str, Enum):
    READ = "READ"
    READ_PROJECT = "READ_PROJECT"
    WRITE_MEMORY = "WRITE_MEMORY"
    WRITE_ARTIFACT = "WRITE_ARTIFACT"
    WRITE_WORKSPACE = "WRITE_WORKSPACE"
    EXECUTE_SAFE = "EXECUTE_SAFE"
    EXECUTE_MEDIA = "EXECUTE_MEDIA"
    WRITE_REPOSITORY = "WRITE_REPOSITORY"
    DESTRUCTIVE = "DESTRUCTIVE"


@dataclass(frozen=True)
class CapabilitySpec:
    name: str
    label: str
    description: str
    cost_class: str
    feature_flag: str
    tools: tuple[str, ...]
    provider: str = "internal"
    permission: str = PermissionClass.READ.value


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
    CapabilitySpec("browser", "Browser", "Read and interact with public JavaScript-rendered pages when fetch is insufficient.", "local_compute", "MCP_PLAYWRIGHT_ENABLED", ("browse_page", "browse_click", "browse_type", "browse_select")),
    CapabilitySpec("time", "Current Time", "Exact current dates, timezones, and time conversion.", "local_compute", "MCP_TIME_ENABLED", ("get_current_time", "convert_time")),
    CapabilitySpec(
        "documentation", "Software Documentation",
        "Use current, version-specific software and framework documentation when it materially improves implementation accuracy.",
        "external", "MCP_CONTEXT7_ENABLED", ("resolve_library_id", "query_documentation"), "context7",
    ),
    CapabilitySpec("github", "GitHub", "Inspect authoritative remote repositories, code, issues, pull requests, commits, and releases.", "external", "MCP_GITHUB_ENABLED", ("github_search_code", "github_get_file", "github_read_commits", "github_read_issues", "github_get_pull_request", "github_read_releases")),
    CapabilitySpec(
        "git", "Local Git",
        "Read repository status, diffs, history, commits, blame, and branch tracking through scoped semantic operations; prefer these over generic command execution for Git reads.",
        "local_compute", "MCP_GIT_ENABLED",
        ("git_status", "git_diff", "git_log", "git_show", "git_blame", "git_branch_info"), "local_git",
    ),
    CapabilitySpec(
        "academic", "Academic Research",
        "Use scholarly identity, publication, citation, DOI, and researcher metadata only when scholarly evidence materially improves the answer.",
        "varies", "MCP_ACADEMIC_ENABLED",
        ("academic_resolve_researcher", "academic_search_publications", "academic_get_researcher_evidence", "academic_compare_source_coverage"),
    ),
    CapabilitySpec(
        "project", "Project Knowledge",
        "Use authorized project memories, files, conversations, and artifacts when durable project context materially helps.",
        "local_compute", "MCP_PROJECT_ENABLED",
        ("project_get_context", "project_search", "project_list_files", "project_read_file", "project_get_memories", "project_save_memory", "project_list_artifacts", "project_save_artifact"),
    ),
    CapabilitySpec(
        "media", "Media", "Plan, generate, edit, adjust, and optionally save images as Project artifacts.",
        "gpu_compute", "MCP_MEDIA_ENABLED",
        ("media_inspect_capability", "media_get_status", "media_generate_image", "media_edit_image", "media_adjust_pose"),
    ),
    CapabilitySpec(
        "google", "Google Workspace",
        "Use Google Drive, Docs, and Sheets for authorized workspace document operations when the user explicitly requests Google Workspace access.",
        "external", "MCP_GOOGLE_ENABLED",
        ("google_drive_list", "google_docs_create", "google_sheets_create", "google_sheets_add_chart"), "google",
    ),
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
    "browse_type": AgentToolSpec("browse_type", "browser", "Open a public page and type bounded non-secret text into one exact field, optionally submitting it. Never enter credentials or sensitive data.", "browser-mcp", "local_compute", "READ", _object({"url": _string("Public HTTPS URL", 2000), "target": _string("Exact snapshot reference or unique selector", 500), "text": _string("Non-secret text to type", 1000), "submit": {"type": "boolean"}, "element": _string("Human-readable field description", 300)}, ("url", "target", "text"))),
    "browse_select": AgentToolSpec("browse_select", "browser", "Open a public page and select bounded values in one exact dropdown. Never use this to alter accounts or make purchases.", "browser-mcp", "local_compute", "READ", _object({"url": _string("Public HTTPS URL", 2000), "target": _string("Exact snapshot reference or unique selector", 500), "values": {"type": "array", "items": _string("Dropdown value", 200), "minItems": 1, "maxItems": 10}, "element": _string("Human-readable dropdown description", 300)}, ("url", "target", "values"))),
    "resolve_library_id": AgentToolSpec("resolve_library_id", "documentation", "Resolve an official library name before querying current Context7 documentation.", "context7-mcp", "external", "READ", _object({"library_name": _string("Official library name", 200), "query": _string("Version-specific documentation need", 1000)}, ("library_name", "query"))),
    "query_documentation": AgentToolSpec("query_documentation", "documentation", "Use this when current library/framework documentation or version-specific implementation details materially improve the answer.", "context7-mcp", "external", "READ", _object({"library_id": _string("Exact Context7 /org/project ID", 300), "query": _string("One specific documentation topic", 1000)}, ("library_id", "query"))),
    "github_search_code": AgentToolSpec("github_search_code", "github", "Search code through the official GitHub MCP server in read-only mode.", "github-mcp", "external", "READ", _object({"query": _string("GitHub code search query", 256)}, ("query",))),
    "github_get_file": AgentToolSpec("github_get_file", "github", "Read one GitHub repository file or directory at an optional ref.", "github-mcp", "external", "READ", _object({"owner": _string("Repository owner", 100), "repo": _string("Repository name", 100), "path": _string("Repository path", 500), "ref": _string("Optional branch, tag, or commit ref", 200)}, ("owner", "repo"))),
    "github_read_commits": AgentToolSpec("github_read_commits", "github", "List bounded remote GitHub commit history or read one commit with file statistics.", "github-mcp", "external", "READ", _object({"owner": _string("Repository owner", 100), "repo": _string("Repository name", 100), "operation": {"type": "string", "enum": ["list", "get"]}, "path": _string("Optional path", 500), "ref": _string("Optional list branch, tag, or commit ref", 200), "sha": _string("Required get SHA, branch, or tag", 200)}, ("owner", "repo", "operation"))),
    "github_read_issues": AgentToolSpec("github_read_issues", "github", "Read one GitHub issue or search bounded issue and pull-request metadata.", "github-mcp", "external", "READ", _object({"operation": {"type": "string", "enum": ["get", "search"]}, "owner": _string("Repository owner for get or scoped search", 100), "repo": _string("Repository name for get or scoped search", 100), "number": {"type": "integer", "minimum": 1}, "query": _string("Issue search query", 256), "include_comments": {"type": "boolean"}}, ("operation",))),
    "github_get_pull_request": AgentToolSpec("github_get_pull_request", "github", "Read one GitHub pull request view, including bounded metadata, files, reviews, checks, status, or diff.", "github-mcp", "external", "READ", _object({"owner": _string("Repository owner", 100), "repo": _string("Repository name", 100), "number": {"type": "integer", "minimum": 1}, "view": {"type": "string", "enum": ["details", "files", "commits", "reviews", "review_comments", "comments", "checks", "status", "diff"]}}, ("owner", "repo", "number"))),
    "github_read_releases": AgentToolSpec("github_read_releases", "github", "List bounded GitHub releases, read the latest release, or read one release by tag.", "github-mcp", "external", "READ", _object({"owner": _string("Repository owner", 100), "repo": _string("Repository name", 100), "operation": {"type": "string", "enum": ["list", "latest", "tag"]}, "tag": _string("Required tag for tag operation", 200)}, ("owner", "repo"))),
    "get_current_time": AgentToolSpec("get_current_time", "time", "Get exact current time in up to five IANA timezones. Use this to resolve relative dates instead of guessing.", "developer-mcp", "local_compute", "READ", _object({"timezones": {"type": "array", "items": _string("IANA timezone", 100), "maxItems": 5}})),
    "convert_time": AgentToolSpec("convert_time", "time", "Convert an ISO date-time between IANA timezones.", "developer-mcp", "local_compute", "READ", _object({"value": _string("ISO 8601 date-time", 100), "from_timezone": _string("Source IANA timezone", 100), "to_timezone": _string("Target IANA timezone", 100)}, ("value", "from_timezone", "to_timezone"))),
    "git_status": AgentToolSpec("git_status", "git", "Read concise repository status. Prefer this scoped semantic operation over generic command execution for Git status.", "developer-mcp", "local_compute", "READ", _object({})),
    "git_log": AgentToolSpec("git_log", "git", "Read bounded local commit history, optionally for one repository-relative path.", "developer-mcp", "local_compute", "READ", _object({"limit": {"type": "integer", "minimum": 1, "maximum": 50}, "relative_path": _string("Optional repository-relative path")})),
    "git_diff": AgentToolSpec("git_diff", "git", "Read local Git diff without modifying the worktree.", "developer-mcp", "local_compute", "READ", _object({"relative_path": _string("Optional repository-relative path"), "staged": {"type": "boolean"}, "summary": {"type": "boolean"}})),
    "git_show": AgentToolSpec("git_show", "git", "Read one local commit, tag, or branch.", "developer-mcp", "local_compute", "READ", _object({"revision": _string("Revision such as HEAD or a commit hash", 200), "relative_path": _string("Optional repository-relative path")})),
    "git_blame": AgentToolSpec("git_blame", "git", "Read line provenance for one repository-relative file.", "developer-mcp", "local_compute", "READ", _object({"relative_path": _string("Repository-relative file path"), "revision": _string("Revision", 200)}, ("relative_path",))),
    "git_branch_info": AgentToolSpec("git_branch_info", "git", "Read local branches and upstream tracking information without changing branches.", "developer-mcp", "local_compute", "READ", _object({})),
    "academic_resolve_researcher": AgentToolSpec("academic_resolve_researcher", "academic", "Resolve who a researcher is from independent identity evidence before treating any provider profile as their corpus.", "academic-mcp", "varies", "READ", _object({"query": _string("Researcher name plus known affiliation, field, alias, or identifier", 500)}, ("query",))),
    "academic_search_publications": AgentToolSpec("academic_search_publications", "academic", "Search a bounded scholarly metadata subset when scholarly evidence materially improves the answer; use Web or Browser later for publisher and full-page verification.", "academic-mcp", "varies", "READ", _object({"query": _string("Focused scholarly publication query", 500), "limit": {"type": "integer", "minimum": 1, "maximum": 10}}, ("query",))),
    "academic_get_researcher_evidence": AgentToolSpec("academic_get_researcher_evidence", "academic", "Get bounded multi-source identity, corpus, representative-paper, and source-specific citation evidence for a researcher evaluation.", "academic-mcp", "varies", "READ", _object({"query": _string("Resolved researcher query with identity hints", 500)}, ("query",))),
    "academic_compare_source_coverage": AgentToolSpec("academic_compare_source_coverage", "academic", "Compare source-specific publication and citation coverage without collapsing conflicts into one metric.", "academic-mcp", "varies", "READ", _object({"query": _string("Researcher query with identity hints", 500)}, ("query",))),
    "project_get_context": AgentToolSpec("project_get_context", "project", "Load a compact relevant project bundle instead of injecting the whole project into context.", "project-mcp", "local_compute", "READ_PROJECT", _object({"query": _string("Current task or retrieval question"), "max_chars": {"type": "integer", "minimum": 1000, "maximum": 12000}}, ("query",))),
    "project_search": AgentToolSpec("project_search", "project", "Search authorized current-project memories, files, conversations, and artifact metadata through the existing project index.", "project-mcp", "local_compute", "READ_PROJECT", _object({"query": _string("Current-project search query")}, ("query",))),
    "project_list_files": AgentToolSpec("project_list_files", "project", "List bounded metadata for files in the authorized current project.", "project-mcp", "local_compute", "READ_PROJECT", _object({})),
    "project_read_file": AgentToolSpec("project_read_file", "project", "Read one bounded chunk of an authorized project file by opaque file ID.", "project-mcp", "local_compute", "READ_PROJECT", _object({"file_id": _string("Opaque project file ID", 100), "offset": {"type": "integer", "minimum": 0}, "max_chars": {"type": "integer", "minimum": 1, "maximum": 12000}}, ("file_id",))),
    "project_get_memories": AgentToolSpec("project_get_memories", "project", "Read active durable memories from the authorized current project; conversation history is not memory.", "project-mcp", "local_compute", "READ_PROJECT", _object({"query": _string("Optional memory search query")})),
    "project_save_memory": AgentToolSpec("project_save_memory", "project", "Save only durable project knowledge that remains useful beyond this exchange, preserving supersession history.", "project-mcp", "local_compute", "WRITE_MEMORY", _object({"memory_type": {"type": "string", "enum": ["fact", "decision", "goal", "constraint", "preference", "todo", "research_result", "summary"]}, "content": _string("Durable non-ephemeral project knowledge", 12000), "confidence": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]}, "supersedes_ids": {"type": "array", "items": _string("Existing project memory ID", 100), "maxItems": 20}}, ("memory_type", "content"))),
    "project_list_artifacts": AgentToolSpec("project_list_artifacts", "project", "List bounded artifact metadata and provenance for the authorized current project.", "project-mcp", "local_compute", "READ_PROJECT", _object({"limit": {"type": "integer", "minimum": 1, "maximum": 200}})),
    "project_save_artifact": AgentToolSpec("project_save_artifact", "project", "Save a bounded text report, document, analysis export, or code output as a project artifact when requested.", "project-mcp", "local_compute", "WRITE_ARTIFACT", _object({"name": _string("Artifact filename only", 160), "content": _string("Artifact text content", 12000), "artifact_type": {"type": "string", "enum": ["report", "document", "analysis", "code", "text"]}, "description": _string("Artifact purpose or provenance", 500)}, ("name", "content"))),
    "media_inspect_capability": AgentToolSpec("media_inspect_capability", "media", "Inspect available visual operations and current limits before choosing an image workflow.", "media-mcp", "gpu_compute", "READ", _object({})),
    "media_get_status": AgentToolSpec("media_get_status", "media", "Get current Media health or one recent job state without retrieving image bytes.", "media-mcp", "gpu_compute", "READ", _object({"job_id": _string("Optional opaque media job ID", 100)})),
    "media_generate_image": AgentToolSpec("media_generate_image", "media", "Generate one image from explicit subject, composition, style, camera, lighting, and background intent; optionally save it to the current Project.", "media-mcp", "gpu_compute", "EXECUTE_MEDIA", _object({"subject": _string("Primary visual subject", 500), "intent": _string("Additional visual intent", 2000), "composition": _string("Composition", 300), "style": _string("Visual style", 300), "camera": _string("Camera or framing", 300), "lighting": _string("Lighting", 300), "background": _string("Background", 300), "quality_priority": {"type": "string", "enum": ["FAST", "BALANCED", "QUALITY"]}, "latency_priority": {"type": "string", "enum": ["FAST", "BALANCED"]}, "output_size": {"type": "string", "enum": ["512x512"]}, "save_to_project": {"type": "boolean"}}, ("subject",))),
    "media_edit_image": AgentToolSpec("media_edit_image", "media", "Apply every requested visual change to one authorized logical source image while preserving identity and unchanged elements.", "media-mcp", "gpu_compute", "EXECUTE_MEDIA", _object({"source_image_id": _string("Opaque generated image or current Project file ID", 100), "intent": _string("Complete edit request; preserve all distinct modifications", 2000), "preserve_identity": {"type": "boolean"}, "preserve_elements": {"type": "array", "items": _string("Element to preserve", 200), "maxItems": 20}, "quality_priority": {"type": "string", "enum": ["FAST", "BALANCED", "QUALITY"]}, "latency_priority": {"type": "string", "enum": ["FAST", "BALANCED"]}, "output_size": {"type": "string", "enum": ["512x512"]}, "save_to_project": {"type": "boolean"}}, ("source_image_id", "intent"))),
    "media_adjust_pose": AgentToolSpec("media_adjust_pose", "media", "Adjust one authorized portrait source image to a front-facing pose while preserving identity.", "media-mcp", "gpu_compute", "EXECUTE_MEDIA", _object({"source_image_id": _string("Opaque generated image or current Project file ID", 100), "pose": {"type": "string", "enum": ["front_facing"]}, "preserve_identity": {"type": "boolean"}, "save_to_project": {"type": "boolean"}}, ("source_image_id",))),
    "google_drive_list": AgentToolSpec("google_drive_list", "google", "List or search files available through the connected user's Google Drive drive.file grant and return bounded metadata only.", "google-mcp", "external", "READ", _object({"query": _string("Optional file-name search text", 300), "mime_type": _string("Optional MIME type filter, e.g. application/vnd.google-apps.document", 200), "limit": {"type": "integer", "minimum": 1, "maximum": 100}, "page_size": {"type": "integer", "minimum": 1, "maximum": 100}, "page_token": _string("Optional token from the preceding page", 2048)})),
    "google_docs_create": AgentToolSpec("google_docs_create", "google", "Create a Google Docs document for the connected user and insert the supplied content as plain text.", "google-mcp", "external", "WRITE_ARTIFACT", _object({"title": _string("Document title", 300), "content": _string("Document body inserted as plain text", 20000), "folder_id": _string("Reserved for future folder placement; omit in this phase", 100)}, ("title", "content"))),
    "google_sheets_create": AgentToolSpec("google_sheets_create", "google", "Create a Google Sheets spreadsheet and write bounded tabular scalar values.", "google-mcp", "external", "WRITE_ARTIFACT", _object({"title": _string("Spreadsheet title", 300), "values": {"type": "array", "items": {"type": "array", "items": {"type": ["string", "number", "boolean", "null"]}, "minItems": 1, "maxItems": 50}, "minItems": 1, "maxItems": 500}, "headers": {"type": "array", "items": {"type": ["string", "number", "boolean", "null"]}, "minItems": 1, "maxItems": 50}, "rows": {"type": "array", "items": {"type": "array", "items": {"type": ["string", "number", "boolean", "null"]}, "minItems": 1, "maxItems": 50}, "maxItems": 499}, "sheet_name": _string("Optional first worksheet name", 100), "start_range": _string("Optional A1 start range; defaults to A1", 100), "folder_id": _string("Reserved for future folder placement; omit in this phase", 100)}, ("title",))),
    "google_sheets_add_chart": AgentToolSpec("google_sheets_add_chart", "google", "Add a supported embedded chart to an existing Google Sheet using a validated A1 data range.", "google-mcp", "external", "WRITE_ARTIFACT", _object({"spreadsheet_id": _string("Target Google spreadsheet ID", 200), "chart_type": {"type": "string", "enum": ["LINE", "BAR", "COLUMN", "PIE"]}, "data_range": _string("A1 range with optional worksheet name, e.g. A1:B5 or Sheet1!A1:B5", 200), "title": _string("Optional chart title", 300), "sheet_id": {"type": "integer", "minimum": 0}, "anchor_row": {"type": "integer", "minimum": 0}, "anchor_column": {"type": "integer", "minimum": 0}}, ("spreadsheet_id", "chart_type", "data_range"))),
}


def _enabled(flag: str, default: bool = True) -> bool:
    value = os.getenv(flag)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def capability_catalog(
    *,
    project_available: bool = False,
    image_available: bool = False,
    google_available: bool | None = None,
) -> list[dict[str, object]]:
    catalog = []
    for capability in CAPABILITIES:
        default = False if capability.name in {"browser", "media", "google"} else True
        if capability.name == "media" and os.getenv(capability.feature_flag) is None:
            enabled = _enabled("MCP_ENABLED", False) and _enabled("MCP_IMAGE_ENABLED", False)
        else:
            enabled = _enabled("MCP_ENABLED", False) and _enabled(capability.feature_flag, default)
        configured = True
        if capability.name == "browser":
            configured = _enabled("MCP_PLAYWRIGHT_EGRESS_GUARD", False)
        elif capability.name == "github":
            configured = bool(os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN"))
        elif capability.name == "project":
            configured = project_available
        elif capability.name == "media":
            configured = image_available
        elif capability.name == "google":
            configured = bool(os.getenv("GOOGLE_WORKSPACE_CREDENTIALS")) if google_available is None else google_available
        provider_states: dict[str, str] | None = None
        health = "AVAILABLE" if enabled and configured else ("UNCONFIGURED" if enabled else "DISABLED")
        if capability.name == "academic" and enabled:
            from runtime.academic_intelligence import academic_source_status

            provider_states = academic_source_status()
            usable = any(status in {"AVAILABLE_FULL", "AVAILABLE_LIMITED"} for status in provider_states.values())
            configured = configured and usable
            if configured and any(status not in {"AVAILABLE_FULL", "AVAILABLE_LIMITED"} for status in provider_states.values()):
                health = "DEGRADED"
            elif not configured:
                health = "UNAVAILABLE"
        entry: dict[str, object] = {
            "name": capability.name,
            "label": capability.label,
            "provider": capability.provider,
            "description": capability.description,
            "cost_class": capability.cost_class,
            "permission": capability.permission,
            "available": enabled and configured,
            "status": "AVAILABLE" if enabled and configured else ("UNCONFIGURED" if enabled else "DISABLED"),
            "health": health,
        }
        if provider_states is not None:
            entry["provider_states"] = provider_states
        catalog.append(entry)
    return catalog


def detailed_tools(selected_capabilities: Iterable[str], limit: int = 10) -> tuple[AgentToolSpec, ...]:
    selected = tuple(dict.fromkeys(selected_capabilities))
    grouped = {
        capability: [
            spec for spec in TOOL_SPECS.values()
            if spec.capability == capability
            and (spec.name != "fetch_page" or _enabled("MCP_FETCH_ENABLED", True))
        ]
        for capability in selected
    }
    bounded_limit = max(1, min(limit, 10))
    ordered: list[AgentToolSpec] = []
    index = 0
    while len(ordered) < bounded_limit:
        added = False
        for capability in selected:
            tools = grouped[capability]
            if index < len(tools):
                ordered.append(tools[index])
                added = True
                if len(ordered) == bounded_limit:
                    break
        if not added:
            break
        index += 1
    return tuple(ordered)


def registry_snapshot() -> dict[str, object]:
    return {
        "capabilities": [asdict(capability) for capability in CAPABILITIES],
        "tools": [asdict(tool) for tool in TOOL_SPECS.values()],
    }
