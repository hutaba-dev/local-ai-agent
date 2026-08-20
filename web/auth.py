"""Local account storage and signed browser sessions for the web UI."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path


USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{2,39}$")
PASSWORD_MIN_LENGTH = 8


@dataclass(frozen=True)
class User:
    username: str
    role: str
    active: bool


class UserStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('admin', 'guest')),
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                )"""
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path)

    @staticmethod
    def _validate(username: str, password: str | None = None) -> None:
        if not USERNAME_PATTERN.fullmatch(username):
            raise ValueError("username must be 3-40 characters using letters, numbers, '.', '_' or '-'")
        if password is not None and len(password) < PASSWORD_MIN_LENGTH:
            raise ValueError(f"password must be at least {PASSWORD_MIN_LENGTH} characters")

    @staticmethod
    def _hash(password: str) -> str:
        salt = secrets.token_bytes(16)
        digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
        return "scrypt$" + base64.urlsafe_b64encode(salt + digest).decode()

    @staticmethod
    def _matches(password: str, stored: str) -> bool:
        try:
            algorithm, encoded = stored.split("$", maxsplit=1)
            payload = base64.urlsafe_b64decode(encoded.encode())
            salt, expected = payload[:16], payload[16:]
            actual = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)
            return algorithm == "scrypt" and hmac.compare_digest(actual, expected)
        except (ValueError, TypeError):
            return False

    def create(self, username: str, password: str, role: str) -> User:
        self._validate(username, password)
        if role not in {"admin", "guest"}:
            raise ValueError("role must be admin or guest")
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO users(username, password_hash, role, active, created_at) VALUES (?, ?, ?, 1, ?)",
                    (username, self._hash(password), role, datetime.now(UTC).isoformat()),
                )
        except sqlite3.IntegrityError as error:
            raise ValueError("username already exists") from error
        return User(username=username, role=role, active=True)

    def authenticate(self, username: str, password: str) -> User | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT username, password_hash, role, active FROM users WHERE username = ?", (username,)
            ).fetchone()
        if row is None or not row[3] or not self._matches(password, row[1]):
            return None
        return User(username=row[0], role=row[2], active=bool(row[3]))

    def set_password(self, username: str, password: str) -> None:
        self._validate(username, password)
        with self._connect() as connection:
            result = connection.execute(
                "UPDATE users SET password_hash = ? WHERE username = ?", (self._hash(password), username)
            )
        if result.rowcount != 1:
            raise ValueError("username does not exist")


class SessionSigner:
    def __init__(self, secret: str, lifetime_minutes: int = 15) -> None:
        self.secret = secret.encode()
        self.lifetime = timedelta(minutes=lifetime_minutes)

    def create(self, user: User, nonce: str | None = None) -> str:
        expires = int((datetime.now(UTC) + self.lifetime).timestamp())
        payload = f"{user.username}:{user.role}:{nonce or secrets.token_urlsafe(16)}:{expires}".encode()
        encoded = base64.urlsafe_b64encode(payload).decode().rstrip("=")
        signature = hmac.new(self.secret, encoded.encode(), hashlib.sha256).hexdigest()
        return f"{encoded}.{signature}"

    def _claims(self, token: str | None) -> tuple[str, str, str, int] | None:
        if not token or "." not in token:
            return None
        encoded, signature = token.rsplit(".", maxsplit=1)
        expected = hmac.new(self.secret, encoded.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return None
        try:
            padded = encoded + "=" * (-len(encoded) % 4)
            username, role, nonce, expires = base64.urlsafe_b64decode(padded).decode().split(":")
            if role not in {"admin", "guest"} or int(expires) < int(datetime.now(UTC).timestamp()):
                return None
            return username, role, nonce, int(expires)
        except (UnicodeDecodeError, ValueError):
            return None

    def verify(self, token: str | None) -> User | None:
        claims = self._claims(token)
        if claims is None:
            return None
        username, role, _nonce, _expires = claims
        return User(username=username, role=role, active=True)

    def renew(self, token: str | None) -> str | None:
        claims = self._claims(token)
        if claims is None:
            return None
        username, role, nonce, _expires = claims
        return self.create(User(username=username, role=role, active=True), nonce=nonce)

    def session_key(self, token: str | None) -> str | None:
        claims = self._claims(token)
        if claims is None:
            return None
        username, role, nonce, _expires = claims
        return f"{username}:{role}:{nonce}"


def configured_user_store() -> UserStore:
    path = Path(os.getenv("WEB_USER_DB", "local-memory/web-users.sqlite3"))
    return UserStore(path)