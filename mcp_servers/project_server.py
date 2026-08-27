"""Request-scoped MCP facade for existing AHNBYS project knowledge."""

from __future__ import annotations

from typing import Any, Protocol

from mcp.server import MCPServer


class ProjectScope(Protocol):
    tools: Any
    owner_id: str
    project_id: str


MAX_PROJECT_OBSERVATION_CHARS = 40_000


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

    @server.tool(description="Search authorized project files, memories, and conversations.", structured_output=True)
    def project_search(query: str) -> dict[str, object]:
        query = query.strip()
        if not query or len(query) > 500:
            raise ValueError("query must contain between 1 and 500 characters")
        result = scope.tools.hybrid_search(scope.owner_id, scope.project_id, query)
        return {"status": "AVAILABLE", "project_id": scope.project_id, "results": _bounded(result)}

    @server.tool(description="List metadata for files in the authorized current project.", structured_output=True)
    def project_list_files() -> dict[str, object]:
        files = scope.tools.project_file_metadata(scope.owner_id, scope.project_id, 100)
        return {"status": "AVAILABLE", "project_id": scope.project_id, "files": _bounded(files)}

    @server.tool(description="Read one authorized project file by opaque file ID. Raw filesystem paths are never accepted.", structured_output=True)
    def project_read_file(file_id: str) -> dict[str, object]:
        if not file_id.startswith("fil_") or len(file_id) > 100:
            raise ValueError("invalid project file ID")
        result = scope.tools.project_file_read(scope.owner_id, scope.project_id, file_id)
        content = str(result.get("content", ""))[:MAX_PROJECT_OBSERVATION_CHARS]
        return {
            "status": "AVAILABLE",
            "project_id": scope.project_id,
            "metadata": result.get("metadata", {}),
            "content": content,
            "truncated": len(str(result.get("content", ""))) > len(content),
        }

    @server.tool(description="Read active memories from the authorized current project.", structured_output=True)
    def project_get_memories(query: str | None = None) -> dict[str, object]:
        if query:
            memories = scope.tools.project_memory_search(scope.owner_id, scope.project_id, query[:500])
        else:
            memories = scope.tools.store.list_memories(scope.owner_id, scope.project_id, active_only=True, limit=50)
        return {"status": "AVAILABLE", "project_id": scope.project_id, "memories": _bounded(memories[:50])}

    return server
