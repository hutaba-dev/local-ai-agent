from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from mcp_servers.google_server import DRIVE_FILE_FIELDS, GoogleToolScope, list_drive_files
from runtime.mcp_host import call_mcp_tool
from web.google_oauth import DRIVE_FILE_SCOPE, GoogleOAuthError, OAuthTokenResponse


class FakeTokenStore:
    def __init__(self, token: object | None) -> None:
        self.token = token
        self.saved: list[tuple[object, ...]] = []

    def google_token(self, username: str) -> object | None:
        return self.token

    def save_google_token(
        self,
        username: str,
        access_token: str,
        refresh_token: str | None,
        expires_at: int,
        scopes: tuple[str, ...],
        token_type: str,
    ) -> None:
        existing_refresh = getattr(self.token, "refresh_token", None)
        self.token = SimpleNamespace(
            access_token=access_token,
            refresh_token=refresh_token or existing_refresh,
            expires_at=expires_at,
            scopes=scopes,
            token_type=token_type,
        )
        self.saved.append((username, access_token, refresh_token, expires_at, scopes, token_type))


class FakeResponse:
    def __init__(self, status_code: int, payload: object) -> None:
        self.status_code = status_code
        self.payload = payload

    def json(self) -> object:
        return self.payload


class FakeAsyncClient:
    def __init__(self, responses: list[FakeResponse], requests: list[dict[str, object]]) -> None:
        self.responses = responses
        self.requests = requests

    async def __aenter__(self) -> "FakeAsyncClient":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    async def get(self, url: str, **kwargs: object) -> FakeResponse:
        self.requests.append({"url": url, **kwargs})
        return self.responses.pop(0)


class GoogleDriveTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.access_token = "test-access-token-never-return"
        self.refresh_token = "test-refresh-token-never-return"
        self.store = FakeTokenStore(SimpleNamespace(
            access_token=self.access_token,
            refresh_token=self.refresh_token,
            expires_at=4_000_000_000,
            scopes=(DRIVE_FILE_SCOPE,),
            token_type="Bearer",
        ))
        self.scope = GoogleToolScope("alice", self.store)
        self.requests: list[dict[str, object]] = []

    async def call(self, *responses: FakeResponse, **arguments: object) -> dict[str, object]:
        queue = list(responses)
        with patch(
            "mcp_servers.google_server.httpx.AsyncClient",
            side_effect=lambda **_kwargs: FakeAsyncClient(queue, self.requests),
        ):
            return await list_drive_files(self.scope, **arguments)

    async def test_connected_user_lists_minimal_metadata(self) -> None:
        result = await self.call(FakeResponse(200, {"files": [{
            "id": "file-1",
            "name": "AHNBYS notes",
            "mimeType": "application/vnd.google-apps.document",
            "modifiedTime": "2026-09-01T00:00:00Z",
            "createdTime": "2026-08-01T00:00:00Z",
            "webViewLink": "https://docs.google.com/document/d/file-1/edit",
            "size": "999999",
            "owners": [{"emailAddress": "private@example.com"}],
        }]}))

        self.assertEqual(result["status"], "AVAILABLE")
        self.assertEqual(result["scope"], DRIVE_FILE_SCOPE)
        self.assertTrue(result["scope_limited"])
        self.assertEqual(set(result["files"][0]), {
            "id", "name", "mimeType", "modifiedTime", "createdTime", "webViewLink",
        })
        self.assertEqual(self.requests[0]["params"]["fields"], DRIVE_FILE_FIELDS)

    async def test_disconnected_user_returns_not_connected_without_http(self) -> None:
        self.store.token = None

        result = await self.call()

        self.assertEqual(result["status"], "NOT_CONNECTED")
        self.assertEqual(self.requests, [])

    async def test_expired_token_refreshes_then_lists_files(self) -> None:
        self.store.token.expires_at = 0
        refreshed = OAuthTokenResponse("new-access", None, 4_000_000_000, (DRIVE_FILE_SCOPE,), "Bearer")
        with patch(
            "mcp_servers.google_server.google_oauth.refresh_access_token",
            AsyncMock(return_value=refreshed),
        ) as refresh:
            result = await self.call(FakeResponse(200, {"files": []}))

        self.assertEqual(result["status"], "AVAILABLE")
        refresh.assert_awaited_once_with(self.refresh_token)
        self.assertEqual(self.store.token.refresh_token, self.refresh_token)
        self.assertEqual(self.requests[0]["headers"]["Authorization"], "Bearer new-access")

    async def test_refresh_failure_is_normalized(self) -> None:
        self.store.token.expires_at = 0
        with patch(
            "mcp_servers.google_server.google_oauth.refresh_access_token",
            AsyncMock(side_effect=GoogleOAuthError("provider secret detail")),
        ):
            result = await self.call()

        self.assertEqual(result["status"], "AUTH_REFRESH_FAILED")
        self.assertNotIn("provider secret detail", json.dumps(result))

    async def test_google_401_refreshes_once_then_normalizes_failure(self) -> None:
        refreshed = OAuthTokenResponse("new-access", None, 4_000_000_000, (DRIVE_FILE_SCOPE,), "Bearer")
        with patch(
            "mcp_servers.google_server.google_oauth.refresh_access_token",
            AsyncMock(return_value=refreshed),
        ) as refresh:
            result = await self.call(FakeResponse(401, {}), FakeResponse(401, {}))

        self.assertEqual(result["status"], "AUTH_REFRESH_FAILED")
        self.assertEqual(len(self.requests), 2)
        refresh.assert_awaited_once_with(self.refresh_token)

    async def test_google_403_is_permission_denied(self) -> None:
        result = await self.call(FakeResponse(403, {"error": {"message": "sensitive provider detail"}}))

        self.assertEqual(result["status"], "PERMISSION_DENIED")
        self.assertNotIn("sensitive provider detail", json.dumps(result))

    async def test_google_429_is_rate_limited(self) -> None:
        result = await self.call(FakeResponse(429, {}))

        self.assertEqual(result["status"], "RATE_LIMITED")

    async def test_google_5xx_is_unavailable(self) -> None:
        result = await self.call(FakeResponse(503, {}))

        self.assertEqual(result["status"], "GOOGLE_API_UNAVAILABLE")

    async def test_pagination_token_is_forwarded_and_returned(self) -> None:
        result = await self.call(
            FakeResponse(200, {"files": [], "nextPageToken": "next-page"}),
            page_token="current-page",
        )

        self.assertEqual(self.requests[0]["params"]["pageToken"], "current-page")
        self.assertEqual(result["next_page_token"], "next-page")

    async def test_page_size_overrides_limit_and_invalid_bounds_do_not_call_http(self) -> None:
        await self.call(FakeResponse(200, {"files": []}), limit=5, page_size=3)
        self.assertEqual(self.requests[0]["params"]["pageSize"], 3)

        self.requests.clear()
        result = await self.call(limit=101)
        self.assertEqual(result["status"], "INVALID_REQUEST")
        self.assertEqual(self.requests, [])

    async def test_search_and_mime_type_are_escaped_into_drive_query(self) -> None:
        await self.call(
            FakeResponse(200, {"files": []}),
            query="AHNBYS's plan",
            mime_type="application/vnd.google-apps.document",
        )

        drive_query = self.requests[0]["params"]["q"]
        self.assertIn("trashed = false", drive_query)
        self.assertIn("name contains 'AHNBYS\\'s plan'", drive_query)
        self.assertIn("mimeType = 'application/vnd.google-apps.document'", drive_query)

    async def test_tokens_and_authorization_header_are_never_returned(self) -> None:
        result = await self.call(FakeResponse(200, {"files": []}))
        serialized = json.dumps(result)

        self.assertNotIn(self.access_token, serialized)
        self.assertNotIn(self.refresh_token, serialized)
        self.assertNotIn("Authorization", serialized)


class GoogleDriveMCPContractTests(unittest.TestCase):
    def test_disconnected_scope_maps_to_not_connected_outcome(self) -> None:
        scope = GoogleToolScope("alice", FakeTokenStore(None))
        with patch.dict(os.environ, {"MCP_ENABLED": "true", "MCP_GOOGLE_ENABLED": "true"}, clear=False):
            outcome = call_mcp_tool("google_drive_list", {"limit": 5}, google_scope=scope)

        self.assertFalse(outcome.success)
        self.assertTrue(outcome.executed)
        self.assertEqual(outcome.status, "NOT_CONNECTED")
        self.assertEqual(outcome.output["files"], [])

    def test_sheets_remains_unconfigured(self) -> None:
        scope = GoogleToolScope("alice", FakeTokenStore(None))
        with patch.dict(os.environ, {"MCP_ENABLED": "true", "MCP_GOOGLE_ENABLED": "true"}, clear=False):
            outcome = call_mcp_tool(
                "google_sheets_create",
                {"title": "Not created", "headers": ["A"], "rows": []},
                google_scope=scope,
            )

        self.assertFalse(outcome.success)
        self.assertEqual(outcome.status, "UNCONFIGURED")


if __name__ == "__main__":
    unittest.main()
