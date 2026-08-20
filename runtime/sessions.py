"""Ephemeral UUID-backed conversation sessions with bounded turn history."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from uuid import uuid4


@dataclass
class Session:
    id: str
    messages: list[dict[str, str]] = field(default_factory=list)
    updated_at: float = field(default_factory=time)


class SessionStore:
    def __init__(self, max_messages: int = 12) -> None:
        self._max_messages = max_messages
        self._sessions: dict[str, Session] = {}

    def create(self) -> Session:
        session = Session(id=str(uuid4()))
        self._sessions[session.id] = session
        return session

    def get_or_create(self, session_id: str | None) -> Session:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]
        return self.create()

    def append(self, session: Session, role: str, content: str) -> None:
        session.messages.append({"role": role, "content": content})
        session.messages = session.messages[-self._max_messages :]
        session.updated_at = time()