"""Minimal, non-persistent long-term memory model shared by agent roles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4


ALLOWED_KINDS = {"preference", "project_fact", "decision", "workflow"}
SENSITIVE_MARKERS = ("api_key", "password", "private key", "token=", "secret=")


def utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    kind: str
    content: str
    tags: tuple[str, ...]
    created_at: datetime
    updated_at: datetime
    source: str
    expires_at: datetime | None = None


class InMemoryMemoryStore:
    """Explicit-save-only store; persistence belongs to a future local adapter."""

    def __init__(self) -> None:
        self._records: dict[str, MemoryRecord] = {}

    def save(self, *, kind: str, content: str, tags: tuple[str, ...], source: str = "user") -> MemoryRecord:
        if kind not in ALLOWED_KINDS:
            raise ValueError(f"unsupported memory kind: {kind}")
        if source not in {"user", "approved_agent_action"}:
            raise ValueError(f"unsupported memory source: {source}")
        if not content.strip():
            raise ValueError("memory content must not be empty")
        if self._contains_sensitive_marker(content):
            raise ValueError("refusing to store potentially sensitive content")

        timestamp = utc_now()
        record = MemoryRecord(
            id=str(uuid4()),
            kind=kind,
            content=content.strip(),
            tags=tuple(tag.strip().lower() for tag in tags if tag.strip()),
            created_at=timestamp,
            updated_at=timestamp,
            source=source,
        )
        self._records[record.id] = record
        return record

    def search(self, query: str) -> list[MemoryRecord]:
        terms = query.lower().split()
        return [
            record
            for record in self._records.values()
            if all(term in f"{record.content} {' '.join(record.tags)}".lower() for term in terms)
        ]

    def delete(self, record_id: str) -> bool:
        return self._records.pop(record_id, None) is not None

    @staticmethod
    def _contains_sensitive_marker(content: str) -> bool:
        normalized = content.lower()
        return any(marker in normalized for marker in SENSITIVE_MARKERS)