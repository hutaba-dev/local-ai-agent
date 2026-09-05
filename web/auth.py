"""Local account storage and signed browser sessions for the web UI."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken


USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{2,39}$")
PASSWORD_MIN_LENGTH = 8
ROLES = frozenset({"admin", "manager", "guest"})


@dataclass(frozen=True)
class User:
    username: str
    role: str
    active: bool


@dataclass(frozen=True)
class GoogleToken:
    access_token: str
    refresh_token: str | None
    expires_at: int
    scopes: tuple[str, ...]
    token_type: str
    updated_at: str


class UserStore:
    def __init__(self, database_path: Path, oauth_secret: str | None = None) -> None:
        self.database_path = database_path
        secret = oauth_secret or os.getenv("WEB_SESSION_SECRET")
        self._oauth_cipher = Fernet(base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest())) if secret else None
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            schema = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'users'"
            ).fetchone()
            if schema is None:
                self._create_table(connection)
            elif "'manager'" not in schema[0]:
                connection.execute("ALTER TABLE users RENAME TO users_before_manager_role")
                self._create_table(connection)
                connection.execute(
                    """INSERT INTO users(username, password_hash, role, active, created_at)
                    SELECT username, password_hash, role, active, created_at FROM users_before_manager_role"""
                )
                connection.execute("DROP TABLE users_before_manager_role")
            self._create_oauth_tables(connection)

    @staticmethod
    def _create_table(connection: sqlite3.Connection) -> None:
        connection.execute(
            """CREATE TABLE users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('admin', 'manager', 'guest')),
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL
            )"""
        )

    @staticmethod
    def _create_oauth_tables(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """CREATE TABLE IF NOT EXISTS google_oauth_states (
                state_hash TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                session_hash TEXT NOT NULL,
                expires_at INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(username) REFERENCES users(username) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_google_oauth_states_expiry
                ON google_oauth_states(expires_at);
            CREATE TABLE IF NOT EXISTS google_oauth_tokens (
                username TEXT PRIMARY KEY,
                access_token TEXT NOT NULL,
                refresh_token TEXT,
                expires_at INTEGER NOT NULL,
                scopes TEXT NOT NULL,
                token_type TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(username) REFERENCES users(username) ON DELETE CASCADE
            );"""
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
        if role not in ROLES:
            raise ValueError("role must be admin, manager or guest")
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

    @staticmethod
    def _digest(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()

    def create_google_oauth_state(
        self, state: str, username: str, session_key: str, expires_at: int
    ) -> None:
        now = datetime.now(UTC)
        with self._connect() as connection:
            connection.execute("DELETE FROM google_oauth_states WHERE expires_at < ?", (int(now.timestamp()),))
            connection.execute(
                "INSERT INTO google_oauth_states(state_hash, username, session_hash, expires_at, created_at) VALUES (?, ?, ?, ?, ?)",
                (self._digest(state), username, self._digest(session_key), expires_at, now.isoformat()),
            )

    def consume_google_oauth_state(
        self, state: str, username: str, session_key: str, now: int | None = None
    ) -> bool:
        current_time = int(datetime.now(UTC).timestamp()) if now is None else now
        state_hash = self._digest(state)
        with self._connect() as connection:
            row = connection.execute(
                """DELETE FROM google_oauth_states WHERE state_hash = ?
                RETURNING username, session_hash, expires_at""",
                (state_hash,),
            ).fetchone()
        return bool(
            row
            and hmac.compare_digest(row[0], username)
            and hmac.compare_digest(row[1], self._digest(session_key))
            and row[2] >= current_time
        )

    def _cipher(self) -> Fernet:
        if self._oauth_cipher is None:
            raise RuntimeError("OAuth token encryption is not configured")
        return self._oauth_cipher

    def save_google_token(
        self,
        username: str,
        access_token: str,
        refresh_token: str | None,
        expires_at: int,
        scopes: tuple[str, ...],
        token_type: str,
    ) -> None:
        cipher = self._cipher()
        encrypted_access = cipher.encrypt(access_token.encode()).decode()
        encrypted_refresh = cipher.encrypt(refresh_token.encode()).decode() if refresh_token else None
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            if encrypted_refresh is None:
                existing = connection.execute(
                    "SELECT refresh_token FROM google_oauth_tokens WHERE username = ?", (username,)
                ).fetchone()
                encrypted_refresh = existing[0] if existing else None
            connection.execute(
                """INSERT INTO google_oauth_tokens(
                    username, access_token, refresh_token, expires_at, scopes, token_type, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(username) DO UPDATE SET
                    access_token = excluded.access_token,
                    refresh_token = excluded.refresh_token,
                    expires_at = excluded.expires_at,
                    scopes = excluded.scopes,
                    token_type = excluded.token_type,
                    updated_at = excluded.updated_at""",
                (username, encrypted_access, encrypted_refresh, expires_at, json.dumps(scopes), token_type, now),
            )

    def google_token(self, username: str) -> GoogleToken | None:
        with self._connect() as connection:
            row = connection.execute(
                """SELECT access_token, refresh_token, expires_at, scopes, token_type, updated_at
                FROM google_oauth_tokens WHERE username = ?""",
                (username,),
            ).fetchone()
        if row is None:
            return None
        cipher = self._cipher()
        try:
            return GoogleToken(
                access_token=cipher.decrypt(row[0].encode()).decode(),
                refresh_token=cipher.decrypt(row[1].encode()).decode() if row[1] else None,
                expires_at=row[2],
                scopes=tuple(json.loads(row[3])),
                token_type=row[4],
                updated_at=row[5],
            )
        except (InvalidToken, UnicodeDecodeError, ValueError, TypeError) as error:
            raise RuntimeError("Stored OAuth token could not be decrypted") from error


class SessionSigner:
    def __init__(
        self,
        secret: str,
        lifetime_minutes: int = 15,
        admin_lifetime_minutes: int = 24 * 60,
        manager_lifetime_minutes: int = 30,
    ) -> None:
        self.secret = secret.encode()
        self.lifetime = timedelta(minutes=lifetime_minutes)
        self.admin_lifetime = timedelta(minutes=admin_lifetime_minutes)
        self.manager_lifetime = timedelta(minutes=manager_lifetime_minutes)

    def lifetime_for(self, role: str) -> timedelta:
        if role == "admin":
            return self.admin_lifetime
        if role == "manager":
            return self.manager_lifetime
        return self.lifetime

    def create(self, user: User, nonce: str | None = None) -> str:
        expires = int((datetime.now(UTC) + self.lifetime_for(user.role)).timestamp())
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
            if role not in ROLES or int(expires) < int(datetime.now(UTC).timestamp()):
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
    return UserStore(path, os.getenv("WEB_SESSION_SECRET"))