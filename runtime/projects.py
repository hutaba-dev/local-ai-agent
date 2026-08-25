"""Persistent project metadata, retrieval, memory, and confined file storage."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Iterable

import httpx


PROJECT_DATA_ROOT = Path(os.getenv("PROJECT_DATA_ROOT", "/srv/local-ai-data"))
PROJECT_DATABASE_PATH = Path(os.getenv("PROJECT_DATABASE_PATH", "/var/lib/local-ai-agent/projects.db"))
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "qwen3.8-27b")
MEMORY_TYPES = frozenset({"fact", "decision", "goal", "constraint", "preference", "todo", "research_result", "summary"})
CONFIDENCE_LEVELS = frozenset({"LOW", "MEDIUM", "HIGH"})
SECRET_PATTERN = re.compile(
    r"(?i)(?:password|passwd|api[_ -]?key|access[_ -]?token|bearer|private[_ -]?key)\s*[:=]|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----|\b(?:sk|ghp|github_pat)_[A-Za-z0-9_-]{16,}\b"
)
WORD_PATTERN = re.compile(r"[\w가-힣]{2,}", re.UNICODE)
LOGGER = logging.getLogger(__name__)


class ProjectError(RuntimeError):
    pass


class ProjectNotFoundError(ProjectError):
    pass


class ProjectStorageOfflineError(ProjectError):
    pass


class ProjectPathError(ProjectError):
    pass


@dataclass(frozen=True)
class StorageStatus:
    online: bool
    mount_point: str
    total_bytes: int
    used_bytes: int
    free_bytes: int


class ProjectStore:
    def __init__(
        self,
        database_path: Path = PROJECT_DATABASE_PATH,
        data_root: Path = PROJECT_DATA_ROOT,
        *,
        require_mount: bool = True,
        client: httpx.Client | None = None,
    ) -> None:
        self.database_path = database_path
        self.data_root = data_root
        self.require_mount = require_mount
        self._client = client or httpx.Client(timeout=90)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    summary TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    archived_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_projects_owner ON projects(owner_id, updated_at DESC);
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    session_id TEXT,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_conversations_project ON conversations(project_id, updated_at DESC);
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    tool_metadata TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);
                CREATE TABLE IF NOT EXISTS files (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
                    original_name TEXT NOT NULL,
                    storage_path TEXT NOT NULL UNIQUE,
                    mime_type TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    index_status TEXT NOT NULL,
                    extracted_text TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_files_project ON files(project_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
                    file_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                    creator TEXT NOT NULL,
                    source_message_id TEXT,
                    description TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    confidence TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    source_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    superseded_by TEXT REFERENCES memories(id)
                );
                CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project_id, active, updated_at DESC);
                CREATE TABLE IF NOT EXISTS memory_sources (
                    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                    source_type TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(memory_id, source_type, source_id)
                );
                CREATE TABLE IF NOT EXISTS project_events (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    conversation_id TEXT REFERENCES conversations(id) ON DELETE SET NULL,
                    event_type TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    details TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_project ON project_events(project_id, created_at DESC);
                CREATE VIRTUAL TABLE IF NOT EXISTS message_fts USING fts5(entity_id UNINDEXED, project_id UNINDEXED, content);
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(entity_id UNINDEXED, project_id UNINDEXED, content);
                CREATE VIRTUAL TABLE IF NOT EXISTS file_chunk_fts USING fts5(entity_id UNINDEXED, project_id UNINDEXED, filename UNINDEXED, content);
                """
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _id(prefix: str) -> str:
        timestamp = int(datetime.now(UTC).timestamp() * 1000).to_bytes(6, "big").hex()
        return f"{prefix}_{timestamp}{secrets.token_hex(10)}"

    def storage_status(self) -> StorageStatus:
        online = self.data_root.is_dir() and (not self.require_mount or os.path.ismount(self.data_root))
        if not online:
            return StorageStatus(False, str(self.data_root), 0, 0, 0)
        usage = shutil.disk_usage(self.data_root)
        return StorageStatus(True, str(self.data_root), usage.total, usage.used, usage.free)

    def require_storage(self) -> None:
        if not self.storage_status().online:
            raise ProjectStorageOfflineError("project storage is offline")

    def _project_root(self, project_id: str) -> Path:
        if not re.fullmatch(r"prj_[a-f0-9]{32}", project_id):
            raise ProjectPathError("invalid project ID")
        data_root = self.data_root.resolve(strict=False)
        projects_root = self.data_root / "projects"
        if projects_root.is_symlink() or not projects_root.resolve(strict=False).is_relative_to(data_root):
            raise ProjectPathError("invalid projects root")
        project_root = projects_root / project_id
        if project_root.is_symlink() or not project_root.resolve(strict=False).is_relative_to(
            projects_root.resolve(strict=False)
        ):
            raise ProjectPathError("invalid project root")
        return project_root

    def confined_path(self, project_id: str, relative_path: str) -> Path:
        relative = PurePosixPath(relative_path)
        if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
            raise ProjectPathError("invalid project-relative path")
        root = self._project_root(project_id).resolve(strict=False)
        candidate = (root / Path(*relative.parts)).resolve(strict=False)
        if not candidate.is_relative_to(root):
            raise ProjectPathError("path escapes project root")
        current = root
        for part in relative.parts[:-1]:
            current /= part
            if current.is_symlink():
                raise ProjectPathError("symlink paths are not allowed")
        return candidate

    def _project_row(self, connection: sqlite3.Connection, owner_id: str, project_id: str) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM projects WHERE id = ? AND owner_id = ? AND archived_at IS NULL",
            (project_id, owner_id),
        ).fetchone()
        if row is None:
            raise ProjectNotFoundError("project not found")
        return row

    def create_project(self, owner_id: str, name: str, description: str = "") -> dict[str, object]:
        self.require_storage()
        clean_name = name.strip()
        if not 1 <= len(clean_name) <= 120:
            raise ValueError("project name must be 1-120 characters")
        project_id = self._id("prj")
        (self.data_root / "projects").mkdir(mode=0o750, parents=True, exist_ok=True)
        root = self._project_root(project_id)
        root.mkdir(mode=0o750, parents=False)
        for directory in ("files", "artifacts", "conversations", "archive"):
            (root / directory).mkdir(mode=0o750)
        now = self._now()
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO projects(id, owner_id, name, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (project_id, owner_id, clean_name, description.strip()[:2000], now, now),
                )
                self._event(connection, project_id, None, "project_created", owner_id, {"name": clean_name})
        except Exception:
            shutil.rmtree(root, ignore_errors=True)
            raise
        return self.get_project(owner_id, project_id)

    def list_projects(self, owner_id: str) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM projects WHERE owner_id = ? AND archived_at IS NULL ORDER BY updated_at DESC",
                (owner_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_project(self, owner_id: str, project_id: str) -> dict[str, object]:
        with self._connect() as connection:
            project = dict(self._project_row(connection, owner_id, project_id))
            project["conversations"] = [
                dict(row) for row in connection.execute(
                    "SELECT * FROM conversations WHERE project_id = ? ORDER BY updated_at DESC", (project_id,)
                ).fetchall()
            ]
        project["storage_online"] = self.storage_status().online
        return project

    def create_conversation(self, owner_id: str, project_id: str, title: str = "New conversation") -> dict[str, object]:
        self.require_storage()
        conversation_id = self._id("cnv")
        now = self._now()
        with self._connect() as connection:
            self._project_row(connection, owner_id, project_id)
            connection.execute(
                "INSERT INTO conversations(id, project_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (conversation_id, project_id, title.strip()[:160] or "New conversation", now, now),
            )
            self._event(connection, project_id, conversation_id, "conversation_created", owner_id, {"title": title})
        return self.get_conversation(owner_id, project_id, conversation_id)

    def get_conversation(self, owner_id: str, project_id: str, conversation_id: str) -> dict[str, object]:
        with self._connect() as connection:
            self._project_row(connection, owner_id, project_id)
            row = connection.execute(
                "SELECT * FROM conversations WHERE id = ? AND project_id = ?", (conversation_id, project_id)
            ).fetchone()
            if row is None:
                raise ProjectNotFoundError("conversation not found")
        return dict(row)

    def list_messages(self, owner_id: str, project_id: str, conversation_id: str) -> list[dict[str, object]]:
        self.get_conversation(owner_id, project_id, conversation_id)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, role, content, tool_metadata, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at",
                (conversation_id,),
            ).fetchall()
        return [dict(row) | {"tool_metadata": json.loads(row["tool_metadata"])} for row in rows]

    def add_message(
        self,
        owner_id: str,
        project_id: str,
        conversation_id: str,
        role: str,
        content: str,
        tool_metadata: Iterable[dict[str, object]] = (),
    ) -> str:
        self.require_storage()
        if role not in {"user", "assistant"}:
            raise ValueError("invalid message role")
        message_id = self._id("msg")
        now = self._now()
        with self._connect() as connection:
            self._project_row(connection, owner_id, project_id)
            if connection.execute(
                "SELECT 1 FROM conversations WHERE id = ? AND project_id = ?", (conversation_id, project_id)
            ).fetchone() is None:
                raise ProjectNotFoundError("conversation not found")
            connection.execute(
                "INSERT INTO messages(id, conversation_id, role, content, tool_metadata, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (message_id, conversation_id, role, content, json.dumps(list(tool_metadata), ensure_ascii=False), now),
            )
            connection.execute(
                "INSERT INTO message_fts(entity_id, project_id, content) VALUES (?, ?, ?)",
                (message_id, project_id, content),
            )
            if role == "user":
                connection.execute(
                    """UPDATE conversations SET title = ? WHERE id = ?
                       AND (title = 'New conversation' OR title GLOB 'Conversation [0-9]*')""",
                    (content.strip().replace("\n", " ")[:80] or "New conversation", conversation_id),
                )
            connection.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
            connection.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (now, project_id))
        return message_id

    def save_file(
        self,
        owner_id: str,
        project_id: str,
        original_name: str,
        content: bytes,
        mime_type: str,
        extracted_text: str = "",
        conversation_id: str | None = None,
        *,
        artifact: bool = False,
        creator: str = "user",
        description: str = "",
        source_message_id: str | None = None,
    ) -> dict[str, object]:
        self.require_storage()
        if not content:
            raise ValueError("file is empty")
        file_id = self._id("fil")
        safe_name = re.sub(r"[^\w. -]", "_", Path(original_name).name, flags=re.UNICODE).strip(" .")[:160] or "file"
        directory = "artifacts" if artifact else "files"
        relative_path = f"{directory}/{file_id}/{safe_name}"
        destination = self.confined_path(project_id, relative_path)
        now = self._now()
        with self._connect() as connection:
            self._project_row(connection, owner_id, project_id)
            if conversation_id is not None and connection.execute(
                "SELECT 1 FROM conversations WHERE id = ? AND project_id = ?", (conversation_id, project_id)
            ).fetchone() is None:
                raise ProjectNotFoundError("conversation not found")
        destination.parent.mkdir(mode=0o750, parents=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.chmod(0o640)
        temporary.replace(destination)
        digest = hashlib.sha256(content).hexdigest()
        index_status = "indexed" if extracted_text.strip() else "not_indexed"
        try:
            with self._connect() as connection:
                connection.execute(
                    """INSERT INTO files(id, project_id, conversation_id, original_name, storage_path, mime_type, size,
                       sha256, index_status, extracted_text, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (file_id, project_id, conversation_id, safe_name, relative_path, mime_type, len(content), digest,
                     index_status, extracted_text[:200_000], now),
                )
                if extracted_text.strip():
                    for chunk in self._chunks(extracted_text):
                        connection.execute(
                            "INSERT INTO file_chunk_fts(entity_id, project_id, filename, content) VALUES (?, ?, ?, ?)",
                            (file_id, project_id, safe_name, chunk),
                        )
                event_type = "artifact_created" if artifact else "file_uploaded"
                self._event(connection, project_id, conversation_id, event_type, creator, {"file_id": file_id, "name": safe_name})
                if artifact:
                    artifact_id = self._id("art")
                    connection.execute(
                        """INSERT INTO artifacts(id, project_id, conversation_id, file_id, creator, source_message_id,
                           description, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (artifact_id, project_id, conversation_id, file_id, creator, source_message_id, description, now),
                    )
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        return self.get_file(owner_id, project_id, file_id)

    @staticmethod
    def _chunks(text: str, size: int = 4000, overlap: int = 400) -> Iterable[str]:
        cleaned = text.replace("\x00", "").strip()
        position = 0
        while position < len(cleaned):
            yield cleaned[position:position + size]
            position += max(1, size - overlap)

    def list_files(self, owner_id: str, project_id: str) -> list[dict[str, object]]:
        with self._connect() as connection:
            self._project_row(connection, owner_id, project_id)
            rows = connection.execute(
                """SELECT files.*, artifacts.id AS artifact_id FROM files LEFT JOIN artifacts ON artifacts.file_id = files.id
                   WHERE files.project_id = ? ORDER BY files.created_at DESC""",
                (project_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_file(self, owner_id: str, project_id: str, file_id: str) -> dict[str, object]:
        with self._connect() as connection:
            self._project_row(connection, owner_id, project_id)
            row = connection.execute(
                "SELECT * FROM files WHERE id = ? AND project_id = ?", (file_id, project_id)
            ).fetchone()
            if row is None:
                raise ProjectNotFoundError("file not found")
        return dict(row)

    def read_file(self, owner_id: str, project_id: str, file_id: str) -> tuple[dict[str, object], bytes]:
        self.require_storage()
        metadata = self.get_file(owner_id, project_id, file_id)
        path = self.confined_path(project_id, str(metadata["storage_path"]))
        if path.is_symlink() or not path.is_file():
            raise ProjectNotFoundError("stored file is unavailable")
        return metadata, path.read_bytes()

    def delete_file(self, owner_id: str, project_id: str, file_id: str) -> None:
        self.require_storage()
        metadata = self.get_file(owner_id, project_id, file_id)
        path = self.confined_path(project_id, str(metadata["storage_path"]))
        with self._connect() as connection:
            connection.execute("DELETE FROM file_chunk_fts WHERE entity_id = ?", (file_id,))
            connection.execute("DELETE FROM files WHERE id = ? AND project_id = ?", (file_id, project_id))
            self._event(connection, project_id, None, "file_deleted", owner_id, {"file_id": file_id, "name": metadata["original_name"]})
        path.unlink(missing_ok=True)
        try:
            path.parent.rmdir()
        except OSError:
            pass

    def move_file(
        self, owner_id: str, project_id: str, file_id: str, destination_relative_path: str
    ) -> dict[str, object]:
        self.require_storage()
        metadata = self.get_file(owner_id, project_id, file_id)
        source = self.confined_path(project_id, str(metadata["storage_path"]))
        destination = self.confined_path(project_id, destination_relative_path)
        if destination.exists():
            raise ValueError("destination already exists")
        if destination.parts[len(self._project_root(project_id).parts)] not in {"files", "artifacts"}:
            raise ProjectPathError("files may only be moved within project files or artifacts")
        destination.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
        source.replace(destination)
        try:
            with self._connect() as connection:
                connection.execute(
                    "UPDATE files SET storage_path = ?, original_name = ? WHERE id = ? AND project_id = ?",
                    (destination_relative_path, destination.name, file_id, project_id),
                )
                self._event(
                    connection, project_id, None, "file_moved", owner_id,
                    {"file_id": file_id, "destination": destination_relative_path},
                )
        except Exception:
            destination.replace(source)
            raise
        return self.get_file(owner_id, project_id, file_id)

    def add_memory(
        self,
        owner_id: str,
        project_id: str,
        memory_type: str,
        content: str,
        confidence: str = "HIGH",
        source_type: str = "user",
        source_id: str | None = None,
        supersedes: Iterable[str] = (),
    ) -> dict[str, object]:
        self.require_storage()
        clean = content.strip()
        if memory_type not in MEMORY_TYPES:
            raise ValueError("invalid memory type")
        if confidence not in CONFIDENCE_LEVELS:
            raise ValueError("invalid confidence")
        if not clean or SECRET_PATTERN.search(clean):
            raise ValueError("memory is empty or may contain a secret")
        now = self._now()
        memory_id = self._id("mem")
        with self._connect() as connection:
            self._project_row(connection, owner_id, project_id)
            duplicate = connection.execute(
                "SELECT * FROM memories WHERE project_id = ? AND type = ? AND content = ? AND active = 1",
                (project_id, memory_type, clean),
            ).fetchone()
            if duplicate is not None:
                return dict(duplicate)
            connection.execute(
                """INSERT INTO memories(id, project_id, type, content, confidence, source_type, source_id,
                   created_at, updated_at, active) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
                (memory_id, project_id, memory_type, clean, confidence, source_type, source_id, now, now),
            )
            connection.execute(
                "INSERT INTO memory_fts(entity_id, project_id, content) VALUES (?, ?, ?)",
                (memory_id, project_id, clean),
            )
            if source_id:
                connection.execute(
                    "INSERT OR IGNORE INTO memory_sources(memory_id, source_type, source_id, created_at) VALUES (?, ?, ?, ?)",
                    (memory_id, source_type, source_id, now),
                )
            for old_id in supersedes:
                connection.execute(
                    "UPDATE memories SET active = 0, superseded_by = ?, updated_at = ? WHERE id = ? AND project_id = ? AND active = 1",
                    (memory_id, now, old_id, project_id),
                )
            self._event(connection, project_id, source_id if source_type == "conversation" else None,
                        "memory_updated", owner_id, {"memory_id": memory_id, "type": memory_type})
        return self.get_memory(owner_id, project_id, memory_id)

    def list_memories(self, owner_id: str, project_id: str, active_only: bool = False) -> list[dict[str, object]]:
        with self._connect() as connection:
            self._project_row(connection, owner_id, project_id)
            clause = "AND active = 1" if active_only else ""
            rows = connection.execute(
                f"SELECT * FROM memories WHERE project_id = ? {clause} ORDER BY active DESC, updated_at DESC",
                (project_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_memory(self, owner_id: str, project_id: str, memory_id: str) -> dict[str, object]:
        with self._connect() as connection:
            self._project_row(connection, owner_id, project_id)
            row = connection.execute(
                "SELECT * FROM memories WHERE id = ? AND project_id = ?", (memory_id, project_id)
            ).fetchone()
            if row is None:
                raise ProjectNotFoundError("memory not found")
        return dict(row)

    def update_memory(self, owner_id: str, project_id: str, memory_id: str, content: str, active: bool = True) -> dict[str, object]:
        self.require_storage()
        clean = content.strip()
        if not clean or SECRET_PATTERN.search(clean):
            raise ValueError("memory is empty or may contain a secret")
        now = self._now()
        with self._connect() as connection:
            self._project_row(connection, owner_id, project_id)
            result = connection.execute(
                "UPDATE memories SET content = ?, active = ?, updated_at = ? WHERE id = ? AND project_id = ?",
                (clean, int(active), now, memory_id, project_id),
            )
            if result.rowcount != 1:
                raise ProjectNotFoundError("memory not found")
            connection.execute("DELETE FROM memory_fts WHERE entity_id = ?", (memory_id,))
            connection.execute(
                "INSERT INTO memory_fts(entity_id, project_id, content) VALUES (?, ?, ?)",
                (memory_id, project_id, clean),
            )
            self._event(connection, project_id, None, "memory_updated", owner_id, {"memory_id": memory_id})
        return self.get_memory(owner_id, project_id, memory_id)

    def delete_memory(self, owner_id: str, project_id: str, memory_id: str) -> None:
        self.require_storage()
        with self._connect() as connection:
            self._project_row(connection, owner_id, project_id)
            result = connection.execute(
                "UPDATE memories SET active = 0, updated_at = ? WHERE id = ? AND project_id = ?",
                (self._now(), memory_id, project_id),
            )
            if result.rowcount != 1:
                raise ProjectNotFoundError("memory not found")
            self._event(connection, project_id, None, "memory_deleted", owner_id, {"memory_id": memory_id})

    def search(self, owner_id: str, project_id: str, query: str, limit: int = 8) -> dict[str, list[dict[str, object]]]:
        with self._connect() as connection:
            self._project_row(connection, owner_id, project_id)
            expression = self._fts_expression(query)
            if not expression:
                return {"memories": [], "files": [], "conversations": []}
            memories = [dict(row) for row in connection.execute(
                """SELECT memories.*, bm25(memory_fts) AS rank FROM memory_fts JOIN memories ON memories.id = memory_fts.entity_id
                   WHERE memory_fts MATCH ? AND memory_fts.project_id = ? AND memories.active = 1 ORDER BY rank LIMIT ?""",
                (expression, project_id, limit),
            ).fetchall()]
            files = [dict(row) for row in connection.execute(
                """SELECT file_chunk_fts.entity_id AS file_id, filename, snippet(file_chunk_fts, 3, '[', ']', ' … ', 24) AS excerpt,
                   bm25(file_chunk_fts) AS rank FROM file_chunk_fts WHERE file_chunk_fts MATCH ? AND project_id = ? ORDER BY rank LIMIT ?""",
                (expression, project_id, limit),
            ).fetchall()]
            conversations = [dict(row) for row in connection.execute(
                """SELECT messages.id, messages.conversation_id, messages.role,
                   snippet(message_fts, 2, '[', ']', ' … ', 24) AS excerpt, bm25(message_fts) AS rank
                   FROM message_fts JOIN messages ON messages.id = message_fts.entity_id
                   WHERE message_fts MATCH ? AND message_fts.project_id = ? ORDER BY rank LIMIT ?""",
                (expression, project_id, limit),
            ).fetchall()]
        return {"memories": memories, "files": files, "conversations": conversations}

    @staticmethod
    def _fts_expression(query: str) -> str:
        words = WORD_PATTERN.findall(query)[:12]
        return " OR ".join(f'"{word.replace(chr(34), chr(34) * 2)}"' for word in words)

    def context(self, owner_id: str, project_id: str, conversation_id: str, query: str) -> str:
        project = self.get_project(owner_id, project_id)
        self.get_conversation(owner_id, project_id, conversation_id)
        results = self.search(owner_id, project_id, query)
        memories = results["memories"] or self.list_memories(owner_id, project_id, active_only=True)[:12]
        with self._connect() as connection:
            recent = connection.execute(
                "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at DESC LIMIT 10",
                (conversation_id,),
            ).fetchall()[::-1]
        sections = [
            f"Project: {project['name']}",
            f"Description: {project['description']}",
            f"Current project summary:\n{project['summary'] or 'No summary yet.'}",
            "Relevant durable memories:\n" + "\n".join(f"- [{item['type']}] {item['content']}" for item in memories[:12]),
            "Relevant file excerpts:\n" + "\n".join(f"- {item['filename']}: {item['excerpt']}" for item in results["files"][:6]),
            "Recent conversation:\n" + "\n".join(f"{row['role']}: {row['content'][:2000]}" for row in recent),
        ]
        return "\n\n".join(sections)[:24_000]

    def process_durable_updates(
        self,
        owner_id: str,
        project_id: str,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
        source_message_id: str | None = None,
    ) -> None:
        try:
            existing = self.list_memories(owner_id, project_id, active_only=True)[:40]
            project = self.get_project(owner_id, project_id)
            instruction = """Extract only durable project information from the completed exchange.
Do not store ordinary conversation, transient requests, private reasoning, credentials, tokens, passwords, or secrets.
Honor requests not to remember something. Preserve conflicts by superseding old memory IDs instead of deleting history.
Return strict JSON with keys: memories (array), summary (string), artifact (object or null).
Each memory has type, content, confidence, supersedes_ids. Allowed types: fact, decision, goal, constraint, preference, todo, research_result, summary.
The summary is a compact incremental current project summary containing goals, status, facts, decisions, constraints, open questions, TODO, and recent important changes.
Set artifact to {filename, description} only when the user explicitly asks to save the current answer as a report/document; otherwise null."""
            payload = {
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": json.dumps({
                        "current_summary": project["summary"],
                        "active_memories": [{"id": item["id"], "type": item["type"], "content": item["content"]} for item in existing],
                        "user": user_message,
                        "assistant_final": assistant_message,
                    }, ensure_ascii=False)},
                ],
                "temperature": 0,
                "max_tokens": 1200,
                "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {"type": "json_object"},
            }
            response = self._client.post(f"{OPENAI_BASE_URL}/chat/completions", json=payload)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            update = json.loads(content)
            for item in update.get("memories", [])[:12]:
                if not isinstance(item, dict):
                    continue
                supersedes = [value for value in item.get("supersedes_ids", []) if isinstance(value, str)]
                memory_type = str(item.get("type", "fact"))
                if memory_type not in MEMORY_TYPES:
                    memory_type = "fact"
                try:
                    self.add_memory(
                        owner_id, project_id, memory_type, str(item.get("content", "")),
                        self._normalize_confidence(item.get("confidence")), "conversation", conversation_id, supersedes,
                    )
                except ValueError as error:
                    LOGGER.warning("Skipped invalid project memory project_id=%s: %s", project_id, error)
            summary = update.get("summary")
            if isinstance(summary, str) and summary.strip():
                with self._connect() as connection:
                    self._project_row(connection, owner_id, project_id)
                    connection.execute(
                        "UPDATE projects SET summary = ?, updated_at = ? WHERE id = ?",
                        (summary.strip()[:12_000], self._now(), project_id),
                    )
            artifact = update.get("artifact")
            if isinstance(artifact, dict) and artifact.get("filename"):
                filename = Path(str(artifact["filename"])).name
                if not filename.lower().endswith(".md"):
                    filename += ".md"
                self.save_file(
                    owner_id, project_id, filename, assistant_message.encode(), "text/markdown",
                    assistant_message, conversation_id, artifact=True, creator="assistant",
                    description=str(artifact.get("description", ""))[:500], source_message_id=source_message_id,
                )
        except Exception:
            LOGGER.exception("Project durable update failed project_id=%s conversation_id=%s", project_id, conversation_id)

    @staticmethod
    def _normalize_confidence(value: object) -> str:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return "HIGH" if value >= 0.8 else "MEDIUM" if value >= 0.5 else "LOW"
        text = str(value or "MEDIUM").strip().upper()
        if text in CONFIDENCE_LEVELS:
            return text
        try:
            numeric = float(text)
        except ValueError:
            return "MEDIUM"
        return "HIGH" if numeric >= 0.8 else "MEDIUM" if numeric >= 0.5 else "LOW"

    def list_events(self, owner_id: str, project_id: str, limit: int = 100) -> list[dict[str, object]]:
        with self._connect() as connection:
            self._project_row(connection, owner_id, project_id)
            rows = connection.execute(
                "SELECT * FROM project_events WHERE project_id = ? ORDER BY created_at DESC LIMIT ?",
                (project_id, min(limit, 200)),
            ).fetchall()
        return [dict(row) | {"details": json.loads(row["details"])} for row in rows]

    def _event(
        self,
        connection: sqlite3.Connection,
        project_id: str,
        conversation_id: str | None,
        event_type: str,
        actor: str,
        details: dict[str, object],
    ) -> None:
        connection.execute(
            "INSERT INTO project_events(id, project_id, conversation_id, event_type, actor, details, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (self._id("evt"), project_id, conversation_id, event_type, actor, json.dumps(details, ensure_ascii=False), self._now()),
        )

    def status_payload(self) -> dict[str, object]:
        return asdict(self.storage_status())
