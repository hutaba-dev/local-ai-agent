"""Owner-scoped project tools; no raw data-root access is exposed."""

from __future__ import annotations

import mimetypes
from pathlib import PurePosixPath

from runtime.projects import ProjectStore


class ProjectTools:
    def __init__(self, store: ProjectStore) -> None:
        self.store = store

    def project_list(self, owner_id: str) -> list[dict[str, object]]:
        return self.store.list_projects(owner_id)

    def project_create(self, owner_id: str, name: str, description: str = "") -> dict[str, object]:
        return self.store.create_project(owner_id, name, description)

    def project_open(self, owner_id: str, project_id: str) -> dict[str, object]:
        return self.store.get_project(owner_id, project_id)

    def project_file_list(self, owner_id: str, project_id: str) -> list[dict[str, object]]:
        return self.store.list_files(owner_id, project_id)

    def project_file_search(self, owner_id: str, project_id: str, query: str) -> list[dict[str, object]]:
        return self.store.search(owner_id, project_id, query)["files"]

    def project_file_read(self, owner_id: str, project_id: str, file_id: str) -> dict[str, object]:
        metadata, content = self.store.read_file(owner_id, project_id, file_id)
        return {"metadata": metadata, "content": content.decode("utf-8", errors="replace")[:40_000]}

    def project_file_create(
        self,
        owner_id: str,
        project_id: str,
        relative_name: str,
        content: str,
        conversation_id: str | None = None,
    ) -> dict[str, object]:
        name = PurePosixPath(relative_name)
        if name.is_absolute() or len(name.parts) != 1:
            raise ValueError("project_file_create accepts a filename, not a path")
        mime_type = mimetypes.guess_type(name.name)[0] or "text/plain"
        return self.store.save_file(
            owner_id, project_id, name.name, content.encode(), mime_type, content, conversation_id
        )

    def project_file_upload(
        self,
        owner_id: str,
        project_id: str,
        filename: str,
        content: bytes,
        mime_type: str,
        extracted_text: str = "",
        conversation_id: str | None = None,
    ) -> dict[str, object]:
        return self.store.save_file(
            owner_id, project_id, filename, content, mime_type, extracted_text, conversation_id
        )

    def project_file_move(
        self,
        owner_id: str,
        project_id: str,
        file_id: str,
        destination_relative_path: str,
    ) -> dict[str, object]:
        return self.store.move_file(owner_id, project_id, file_id, destination_relative_path)

    def project_artifact_save(
        self,
        owner_id: str,
        project_id: str,
        filename: str,
        content: bytes,
        mime_type: str,
        conversation_id: str,
        description: str = "",
        source_message_id: str | None = None,
    ) -> dict[str, object]:
        extracted = content.decode("utf-8", errors="replace") if mime_type.startswith("text/") else ""
        return self.store.save_file(
            owner_id,
            project_id,
            filename,
            content,
            mime_type,
            extracted,
            conversation_id,
            artifact=True,
            creator="assistant",
            description=description,
            source_message_id=source_message_id,
        )

    def project_memory_search(self, owner_id: str, project_id: str, query: str) -> list[dict[str, object]]:
        return self.store.search(owner_id, project_id, query)["memories"]

    def project_memory_add(
        self,
        owner_id: str,
        project_id: str,
        memory_type: str,
        content: str,
        confidence: str = "HIGH",
        source_id: str | None = None,
    ) -> dict[str, object]:
        return self.store.add_memory(
            owner_id, project_id, memory_type, content, confidence, "conversation", source_id
        )

    def project_memory_update(
        self,
        owner_id: str,
        project_id: str,
        memory_id: str,
        content: str,
        active: bool = True,
    ) -> dict[str, object]:
        return self.store.update_memory(owner_id, project_id, memory_id, content, active)

    def project_conversation_search(
        self, owner_id: str, project_id: str, query: str
    ) -> list[dict[str, object]]:
        return self.store.search(owner_id, project_id, query)["conversations"]

    def semantic_search(self, owner_id: str, project_id: str, query: str) -> dict[str, object]:
        self.store.get_project(owner_id, project_id)
        return {"available": False, "query": query, "results": []}

    def hybrid_search(self, owner_id: str, project_id: str, query: str) -> dict[str, object]:
        return {
            "semantic_available": False,
            "lexical": self.store.search(owner_id, project_id, query),
            "semantic": [],
        }
