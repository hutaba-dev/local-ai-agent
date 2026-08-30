"""Request-scoped MCP facade for existing AHNBYS project knowledge."""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Literal, Protocol
from urllib.parse import unquote

from mcp.server import MCPServer

from runtime.projects import ProjectStorageOfflineError


class ProjectScope(Protocol):
    tools: Any
    owner_id: str
    project_id: str
    conversation_id: str | None


MAX_PROJECT_OBSERVATION_CHARS = 12_000


def _bounded(value: object) -> object:
    remaining = [MAX_PROJECT_OBSERVATION_CHARS]

    def walk(item: object) -> object:
        if remaining[0] <= 0:
            return "[truncated]"
        if isinstance(item, str):
            text = item[:remaining[0]]
            remaining[0] -= len(text)
            return text
        if isinstance(item, list):
            return [walk(child) for child in item[:100] if remaining[0] > 0]
        if isinstance(item, dict):
            return {str(key): walk(child) for key, child in list(item.items())[:100] if remaining[0] > 0}
        remaining[0] -= len(str(item))
        return item

    return walk(value)


def create_project_mcp(scope: ProjectScope) -> MCPServer:
    server = MCPServer(
        "ahnbys-project",
        description="Authenticated project files and memories. Request scope is injected by the host, never selected by the model.",
        version="1.0.0",
    )

    def ready() -> dict[str, object] | None:
        scope.tools.store.get_project(scope.owner_id, scope.project_id)
        try:
            scope.tools.store.require_storage()
        except ProjectStorageOfflineError:
            return {"status": "PROJECT_STORAGE_OFFLINE", "project_id": scope.project_id}
        return None

    def file_metadata(value: object) -> dict[str, object]:
        if not isinstance(value, dict):
            return {}
        allowed = {
            "id", "project_id", "conversation_id", "original_name", "mime_type", "size", "sha256",
            "index_status", "created_at", "artifact_id",
        }
        return {key: item for key, item in value.items() if key in allowed}

    @server.tool(
        description="Load a compact authorized project bundle: summary, relevant durable memories, file/conversation excerpts, and artifact metadata.",
        structured_output=True,
    )
    def project_get_context(query: str, max_chars: int = 10_000) -> dict[str, object]:
        unavailable = ready()
        if unavailable:
            return unavailable
        query = query.strip()
        if not query or len(query) > 500:
            raise ValueError("query must contain between 1 and 500 characters")
        result = scope.tools.project_get_context(scope.owner_id, scope.project_id, query, max_chars)
        return {"status": "AVAILABLE", "project_id": scope.project_id, "context": _bounded(result)}

    @server.tool(description="Search authorized project files, memories, and conversations.", structured_output=True)
    def project_search(query: str) -> dict[str, object]:
        unavailable = ready()
        if unavailable:
            return unavailable
        query = query.strip()
        if not query or len(query) > 500:
            raise ValueError("query must contain between 1 and 500 characters")
        result = scope.tools.hybrid_search(scope.owner_id, scope.project_id, query)
        return {"status": "AVAILABLE", "project_id": scope.project_id, "results": _bounded(result)}

    @server.tool(description="List metadata for files in the authorized current project.", structured_output=True)
    def project_list_files() -> dict[str, object]:
        unavailable = ready()
        if unavailable:
            return unavailable
        files = scope.tools.project_file_metadata(scope.owner_id, scope.project_id, 100)
        return {"status": "AVAILABLE", "project_id": scope.project_id, "files": _bounded([file_metadata(item) for item in files])}

    @server.tool(description="Read one authorized project file by opaque file ID. Raw filesystem paths are never accepted.", structured_output=True)
    def project_read_file(file_id: str, offset: int = 0, max_chars: int = 12_000) -> dict[str, object]:
        unavailable = ready()
        if unavailable:
            return unavailable
        if not file_id.startswith("fil_") or len(file_id) > 100:
            raise ValueError("invalid project file ID")
        result = scope.tools.project_file_read(scope.owner_id, scope.project_id, file_id, offset, max_chars)
        return {
            "status": "AVAILABLE",
            "project_id": scope.project_id,
            "metadata": file_metadata(result.get("metadata")),
            "content": result.get("content", ""),
            "offset": result.get("offset", 0),
            "next_offset": result.get("next_offset"),
            "truncated": result.get("truncated", False),
        }

    @server.tool(description="Read active memories from the authorized current project.", structured_output=True)
    def project_get_memories(query: str | None = None) -> dict[str, object]:
        unavailable = ready()
        if unavailable:
            return unavailable
        if query:
            memories = scope.tools.project_memory_search(scope.owner_id, scope.project_id, query[:500])
        else:
            memories = scope.tools.store.list_memories(scope.owner_id, scope.project_id, active_only=True, limit=50)
        return {"status": "AVAILABLE", "project_id": scope.project_id, "memories": _bounded(memories[:50])}

    @server.tool(
        description="Save one durable, project-scoped memory only when it will remain useful beyond the current exchange. Existing memory IDs may be superseded without deleting history.",
        structured_output=True,
    )
    def project_save_memory(
        memory_type: Literal["fact", "decision", "goal", "constraint", "preference", "todo", "research_result", "summary"],
        content: str,
        confidence: Literal["LOW", "MEDIUM", "HIGH"] = "HIGH",
        supersedes_ids: list[str] | None = None,
    ) -> dict[str, object]:
        unavailable = ready()
        if unavailable:
            return unavailable
        if not 1 <= len(content.strip()) <= 12_000:
            raise ValueError("memory content must contain between 1 and 12000 characters")
        supersedes = tuple(supersedes_ids or ())
        if len(supersedes) > 20 or any(not item.startswith("mem_") or len(item) > 100 for item in supersedes):
            raise ValueError("invalid superseded memory IDs")
        memory = scope.tools.project_memory_add(
            scope.owner_id, scope.project_id, memory_type, content, confidence,
            getattr(scope, "conversation_id", None), supersedes,
        )
        return {"status": "AVAILABLE", "project_id": scope.project_id, "memory": _bounded(memory)}

    @server.tool(description="List bounded metadata for artifacts in the authorized current project.", structured_output=True)
    def project_list_artifacts(limit: int = 50) -> dict[str, object]:
        unavailable = ready()
        if unavailable:
            return unavailable
        artifacts = scope.tools.project_artifact_list(scope.owner_id, scope.project_id, limit)
        return {"status": "AVAILABLE", "project_id": scope.project_id, "artifacts": _bounded(artifacts)}

    @server.tool(
        description="Save a bounded text result as a durable project artifact when the user requests a report, document, analysis export, or code output.",
        structured_output=True,
    )
    def project_save_artifact(
        name: str,
        content: str,
        artifact_type: Literal["report", "document", "analysis", "code", "text"] = "report",
        description: str = "",
    ) -> dict[str, object]:
        unavailable = ready()
        if unavailable:
            return unavailable
        filename = PurePosixPath(unquote(name))
        if filename.is_absolute() or len(filename.parts) != 1 or not 1 <= len(filename.name) <= 160:
            raise ValueError("artifact name must be a filename, not a path")
        if not 1 <= len(content) <= 12_000:
            raise ValueError("artifact content must contain between 1 and 12000 characters")
        mime_type = "text/markdown" if filename.suffix.lower() in {".md", ".markdown"} else "text/plain"
        artifact = scope.tools.project_artifact_save(
            scope.owner_id, scope.project_id, filename.name, content.encode(), mime_type,
            getattr(scope, "conversation_id", None), description[:500],
        )
        return {
            "status": "AVAILABLE",
            "project_id": scope.project_id,
            "artifact_type": artifact_type,
            "artifact": file_metadata(artifact),
        }

    return server
