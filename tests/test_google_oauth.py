from __future__ import annotations

import os
import logging
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from time import time
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient

from web import app as web_app
from web.auth import UserStore
from web.google_oauth import DRIVE_FILE_SCOPE, GoogleOAuthError, OAuthTokenResponse


class GoogleOAuthStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.database = Path(self.temporary_directory.name) / "users.sqlite3"
        self.store = UserStore(self.database, "test-encryption-secret")
        self.store.create("test-admin", "a-test-password", "admin")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_state_is_bound_to_user_and_session_and_single_use(self) -> None:
        self.store.create_google_oauth_state("raw-state", "test-admin", "session-a", 200)

        self.assertFalse(self.store.consume_google_oauth_state("raw-state", "test-admin", "session-b", now=100))
        self.assertFalse(self.store.consume_google_oauth_state("raw-state", "test-admin", "session-a", now=100))

    def test_expired_state_is_rejected_and_consumed(self) -> None:
        self.store.create_google_oauth_state("expired-state", "test-admin", "session-a", 99)

        self.assertFalse(self.store.consume_google_oauth_state("expired-state", "test-admin", "session-a", now=100))
        self.assertFalse(self.store.consume_google_oauth_state("expired-state", "test-admin", "session-a", now=98))

    def test_valid_state_is_single_use_and_raw_state_is_not_stored(self) -> None:
        self.store.create_google_oauth_state("valid-state", "test-admin", "session-a", 200)

        with sqlite3.connect(self.database) as connection:
            stored = connection.execute("SELECT state_hash FROM google_oauth_states").fetchone()[0]
        self.assertNotEqual(stored, "valid-state")
        self.assertTrue(self.store.consume_google_oauth_state("valid-state", "test-admin", "session-a", now=100))
        self.assertFalse(self.store.consume_google_oauth_state("valid-state", "test-admin", "session-a", now=100))

    def test_tokens_are_encrypted_and_refresh_token_is_preserved(self) -> None:
        self.store.save_google_token(
            "test-admin", "access-one", "refresh-one", 200, ("scope-a",), "Bearer"
        )
        self.store.save_google_token(
            "test-admin", "access-two", None, 300, ("scope-a",), "Bearer"
        )

        token = self.store.google_token("test-admin")
        self.assertIsNotNone(token)
        self.assertEqual(token.access_token, "access-two")
        self.assertEqual(token.refresh_token, "refresh-one")
        with sqlite3.connect(self.database) as connection:
            stored = connection.execute(
                "SELECT access_token, refresh_token FROM google_oauth_tokens WHERE username = ?", ("test-admin",)
            ).fetchone()
        self.assertNotIn("access-two", stored[0])
        self.assertNotIn("refresh-one", stored[1])


@patch.dict(os.environ, {
    "MCP_GOOGLE_ENABLED": "1",
    "GOOGLE_CLIENT_ID": "test-client-id",
    "GOOGLE_CLIENT_SECRET": "test-client-secret",
})
class GoogleOAuthRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.previous_user_store = web_app.user_store
        web_app.user_store = UserStore(
            Path(self.temporary_directory.name) / "users.sqlite3", "test-encryption-secret"
        )
        web_app.user_store.create("test-admin", "a-test-password", "admin")
        web_app.user_store.create("test-manager", "a-test-password", "manager")

    def tearDown(self) -> None:
        web_app.user_store = self.previous_user_store
        self.temporary_directory.cleanup()

    @staticmethod
    def authenticated_client(username: str = "test-admin") -> TestClient:
        client = TestClient(web_app.app)
        response = client.post("/api/login", json={"username": username, "password": "a-test-password"})
        if response.status_code != 200:
            raise AssertionError("test login failed")
        return client

    def connect_state(self, client: TestClient) -> str:
        response = client.post("/api/google/connect", follow_redirects=False)
        self.assertEqual(response.status_code, 303)
        location = response.headers["location"]
        query = parse_qs(urlparse(location).query)
        self.assertEqual(query["client_id"], ["test-client-id"])
        self.assertEqual(query["scope"], [DRIVE_FILE_SCOPE])
        self.assertNotIn("test-client-secret", location)
        return query["state"][0]

    def test_google_routes_require_authentication(self) -> None:
        client = TestClient(web_app.app)

        self.assertEqual(client.post("/api/google/connect", follow_redirects=False).status_code, 401)
        self.assertEqual(client.get("/api/google/status", follow_redirects=False).status_code, 401)
        callback = client.get("/oauth/google?state=x&code=y", follow_redirects=False)
        self.assertEqual(callback.status_code, 303)
        self.assertEqual(callback.headers["location"], "/login")

    def test_oauth_callback_query_is_redacted_from_access_log(self) -> None:
        record = logging.LogRecord(
            "uvicorn.access",
            logging.INFO,
            "",
            0,
            '%s - "%s %s HTTP/%s" %d',
            ("127.0.0.1", "GET", "/oauth/google?state=secret-state&code=secret-code", "1.1", 303),
            None,
        )

        self.assertTrue(web_app.OAuthAccessLogFilter().filter(record))
        self.assertEqual(record.args[2], "/oauth/google?<redacted>")

    def test_callback_requires_code_and_valid_current_session_state(self) -> None:
        owner = self.authenticated_client()
        other_browser = self.authenticated_client()
        state = self.connect_state(owner)

        self.assertEqual(owner.get(f"/oauth/google?state={state}", follow_redirects=False).status_code, 400)
        state = self.connect_state(owner)
        self.assertEqual(other_browser.get(f"/oauth/google?state={state}&code=code", follow_redirects=False).status_code, 400)
        self.assertEqual(owner.get(f"/oauth/google?state={state}&code=code", follow_redirects=False).status_code, 400)

    def test_callback_exchanges_and_persists_user_token_without_exposing_it(self) -> None:
        client = self.authenticated_client()
        state = self.connect_state(client)
        token = OAuthTokenResponse(
            "access-secret", "refresh-secret", int(time()) + 3600, (DRIVE_FILE_SCOPE,), "Bearer"
        )

        with patch.object(web_app.google_oauth, "exchange_code", AsyncMock(return_value=token)) as exchange:
            response = client.get(f"/oauth/google?state={state}&code=one-time-code", follow_redirects=False)

        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["location"], "/?google=connected")
        exchange.assert_awaited_once_with("one-time-code")
        stored = web_app.user_store.google_token("test-admin")
        self.assertEqual(stored.access_token, "access-secret")
        status = client.get("/api/google/status")
        self.assertEqual(status.status_code, 200)
        self.assertTrue(status.json()["connected"])
        self.assertNotIn("access-secret", status.text)
        self.assertNotIn("refresh-secret", status.text)
        other_user_status = self.authenticated_client("test-manager").get("/api/google/status")
        self.assertFalse(other_user_status.json()["connected"])

    def test_exchange_failure_is_generic_and_state_cannot_be_reused(self) -> None:
        client = self.authenticated_client()
        state = self.connect_state(client)

        with patch.object(
            web_app.google_oauth, "exchange_code", AsyncMock(side_effect=GoogleOAuthError("secret detail"))
        ):
            response = client.get(f"/oauth/google?state={state}&code=bad", follow_redirects=False)
        self.assertEqual(response.status_code, 502)
        self.assertNotIn("secret detail", response.text)
        self.assertEqual(client.get(f"/oauth/google?state={state}&code=again", follow_redirects=False).status_code, 400)

    def test_status_refreshes_expired_token_and_preserves_refresh_token(self) -> None:
        client = self.authenticated_client()
        web_app.user_store.save_google_token(
            "test-admin", "old-access", "refresh-secret", int(time()) - 1, (DRIVE_FILE_SCOPE,), "Bearer"
        )
        refreshed = OAuthTokenResponse("new-access", None, int(time()) + 3600, (DRIVE_FILE_SCOPE,), "Bearer")

        with patch.object(
            web_app.google_oauth, "refresh_access_token", AsyncMock(return_value=refreshed)
        ) as refresh:
            response = client.get("/api/google/status")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["connected"])
        refresh.assert_awaited_once_with("refresh-secret")
        stored = web_app.user_store.google_token("test-admin")
        self.assertEqual(stored.access_token, "new-access")
        self.assertEqual(stored.refresh_token, "refresh-secret")

    @patch.dict(os.environ, {"MCP_GOOGLE_ENABLED": "0"})
    def test_status_reports_unconfigured_without_connection_details(self) -> None:
        client = self.authenticated_client()

        self.assertEqual(client.get("/api/google/status").json(), {
            "configured": False, "connected": False, "expires_at": None, "scopes": []
        })
        self.assertEqual(client.post("/api/google/connect").status_code, 503)


if __name__ == "__main__":
    unittest.main()