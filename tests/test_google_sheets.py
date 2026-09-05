from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from mcp_servers.google_server import SHEETS_CREATE_ENDPOINT, GoogleToolScope, create_google_spreadsheet
from web.google_oauth import DRIVE_FILE_SCOPE, GoogleOAuthError, OAuthTokenResponse


class FakeTokenStore:
    def __init__(self, token: object | None) -> None:
        self.token = token

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
        self.token = SimpleNamespace(
            access_token=access_token,
            refresh_token=refresh_token or getattr(self.token, "refresh_token", None),
            expires_at=expires_at,
            scopes=scopes,
            token_type=token_type,
        )


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

    async def post(self, url: str, **kwargs: object) -> FakeResponse:
        self.requests.append({"method": "POST", "url": url, **kwargs})
        return self.responses.pop(0)

    async def put(self, url: str, **kwargs: object) -> FakeResponse:
        self.requests.append({"method": "PUT", "url": url, **kwargs})
        return self.responses.pop(0)


class GoogleSheetsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.access_token = "sheets-access-token-never-return"
        self.refresh_token = "sheets-refresh-token-never-return"
        self.store = FakeTokenStore(SimpleNamespace(
            access_token=self.access_token,
            refresh_token=self.refresh_token,
            expires_at=4_000_000_000,
            scopes=(DRIVE_FILE_SCOPE,),
            token_type="Bearer",
        ))
        self.scope = GoogleToolScope("alice", self.store)
        self.requests: list[dict[str, object]] = []
        self.values: list[list[object]] = [["Name", "Count", "Ready"], ["alpha", 2, True], ["beta", None, False]]

    async def call(self, *responses: FakeResponse, **arguments: object) -> dict[str, object]:
        queue = list(responses)
        with patch(
            "mcp_servers.google_server.httpx.AsyncClient",
            side_effect=lambda **_kwargs: FakeAsyncClient(queue, self.requests),
        ):
            return await create_google_spreadsheet(self.scope, **arguments)

    async def test_create_and_values_write_return_id_url_and_dimensions(self) -> None:
        result = await self.call(
            FakeResponse(200, {"spreadsheetId": "sheet-123"}),
            FakeResponse(200, {"updatedRows": 3, "updatedColumns": 3}),
            title="AHNBYS table",
            values=self.values,
            sheet_name="Results",
        )

        self.assertEqual(result["status"], "AVAILABLE")
        self.assertEqual(result["spreadsheet_id"], "sheet-123")
        self.assertEqual(result["url"], "https://docs.google.com/spreadsheets/d/sheet-123/edit")
        self.assertEqual(result["rows_written"], 3)
        self.assertEqual(result["columns_written"], 3)
        self.assertEqual(result["scope"], DRIVE_FILE_SCOPE)
        self.assertTrue(result["scope_limited"])
        self.assertEqual(self.requests[0]["url"], SHEETS_CREATE_ENDPOINT)
        self.assertEqual(self.requests[0]["json"], {
            "properties": {"title": "AHNBYS table"},
            "sheets": [{"properties": {"title": "Results"}}],
        })
        self.assertEqual(self.requests[1]["method"], "PUT")
        self.assertTrue(str(self.requests[1]["url"]).endswith("/values/%27Results%27%21A1"))
        self.assertEqual(self.requests[1]["params"], {"valueInputOption": "RAW"})
        self.assertEqual(self.requests[1]["json"], {
            "range": "'Results'!A1",
            "majorDimension": "ROWS",
            "values": self.values,
        })

    async def test_legacy_headers_and_rows_are_combined(self) -> None:
        await self.call(
            FakeResponse(200, {"spreadsheetId": "legacy"}),
            FakeResponse(200, {}),
            title="Legacy",
            headers=["A", "B"],
            rows=[[1, 2]],
        )

        self.assertEqual(self.requests[1]["json"]["values"], [["A", "B"], [1, 2]])

    async def test_disconnected_user_returns_not_connected_without_http(self) -> None:
        self.store.token = None
        result = await self.call(title="Title", values=[["A"]])
        self.assertEqual(result["status"], "NOT_CONNECTED")
        self.assertEqual(self.requests, [])

    async def test_expired_token_refreshes_and_preserves_refresh_token(self) -> None:
        self.store.token.expires_at = 0
        refreshed = OAuthTokenResponse("new-access", None, 4_000_000_000, (DRIVE_FILE_SCOPE,), "Bearer")
        with patch(
            "mcp_servers.google_server.google_oauth.refresh_access_token",
            AsyncMock(return_value=refreshed),
        ) as refresh:
            result = await self.call(
                FakeResponse(200, {"spreadsheetId": "refreshed"}),
                FakeResponse(200, {}),
                title="Title",
                values=[["A"]],
            )

        self.assertEqual(result["status"], "AVAILABLE")
        refresh.assert_awaited_once_with(self.refresh_token)
        self.assertEqual(self.store.token.refresh_token, self.refresh_token)
        self.assertEqual(self.requests[0]["headers"]["Authorization"], "Bearer new-access")

    async def test_refresh_failure_is_normalized(self) -> None:
        self.store.token.expires_at = 0
        with patch(
            "mcp_servers.google_server.google_oauth.refresh_access_token",
            AsyncMock(side_effect=GoogleOAuthError("provider detail")),
        ):
            result = await self.call(title="Title", values=[["A"]])

        self.assertEqual(result["status"], "AUTH_REFRESH_FAILED")
        self.assertNotIn("provider detail", json.dumps(result))

    async def test_create_401_refreshes_once_then_succeeds(self) -> None:
        refreshed = OAuthTokenResponse("new-access", None, 4_000_000_000, (DRIVE_FILE_SCOPE,), "Bearer")
        with patch(
            "mcp_servers.google_server.google_oauth.refresh_access_token",
            AsyncMock(return_value=refreshed),
        ) as refresh:
            result = await self.call(
                FakeResponse(401, {}),
                FakeResponse(200, {"spreadsheetId": "retried"}),
                FakeResponse(200, {}),
                title="Title",
                values=[["A"]],
            )

        self.assertEqual(result["status"], "AVAILABLE")
        self.assertEqual(len(self.requests), 3)
        refresh.assert_awaited_once_with(self.refresh_token)

    async def test_create_403_is_permission_denied(self) -> None:
        result = await self.call(FakeResponse(403, {"private": "detail"}), title="Title", values=[["A"]])
        self.assertEqual(result["status"], "PERMISSION_DENIED")
        self.assertNotIn("detail", json.dumps(result))

    async def test_create_429_is_rate_limited(self) -> None:
        result = await self.call(FakeResponse(429, {}), title="Title", values=[["A"]])
        self.assertEqual(result["status"], "RATE_LIMITED")

    async def test_create_5xx_is_unavailable(self) -> None:
        result = await self.call(FakeResponse(503, {}), title="Title", values=[["A"]])
        self.assertEqual(result["status"], "GOOGLE_API_UNAVAILABLE")

    async def test_missing_spreadsheet_id_is_create_failed(self) -> None:
        result = await self.call(FakeResponse(200, {}), title="Title", values=[["A"]])
        self.assertEqual(result["status"], "SPREADSHEET_CREATE_FAILED")

    async def test_values_401_refreshes_once_then_succeeds(self) -> None:
        refreshed = OAuthTokenResponse("new-access", None, 4_000_000_000, (DRIVE_FILE_SCOPE,), "Bearer")
        with patch(
            "mcp_servers.google_server.google_oauth.refresh_access_token",
            AsyncMock(return_value=refreshed),
        ) as refresh:
            result = await self.call(
                FakeResponse(200, {"spreadsheetId": "write-retried"}),
                FakeResponse(401, {}),
                FakeResponse(200, {}),
                title="Title",
                values=[["A"]],
            )

        self.assertEqual(result["status"], "AVAILABLE")
        self.assertEqual(len(self.requests), 3)
        refresh.assert_awaited_once_with(self.refresh_token)
        self.assertEqual(self.requests[-1]["headers"]["Authorization"], "Bearer new-access")

    async def test_values_write_failure_reports_partial_spreadsheet(self) -> None:
        result = await self.call(
            FakeResponse(200, {"spreadsheetId": "partial"}),
            FakeResponse(400, {"error": "private write detail"}),
            title="Title",
            values=[["A"]],
        )

        self.assertEqual(result["status"], "VALUES_WRITE_FAILED")
        self.assertEqual(result["spreadsheet_id"], "partial")
        self.assertTrue(result["partial"])
        self.assertNotIn("private write detail", json.dumps(result))

    async def test_invalid_values_are_rejected_without_http(self) -> None:
        cases: list[object] = [[], ["not-a-row"], [[]], [[{"nested": True}]], [[float("inf")]]]
        for values in cases:
            with self.subTest(values=values):
                result = await self.call(title="Title", values=values)
                self.assertEqual(result["status"], "INVALID_REQUEST")
        self.assertEqual(self.requests, [])

    async def test_empty_title_is_rejected_without_http(self) -> None:
        result = await self.call(title=" ", values=[["A"]])
        self.assertEqual(result["status"], "INVALID_REQUEST")
        self.assertEqual(self.requests, [])

    async def test_oversized_values_are_rejected_without_http(self) -> None:
        result = await self.call(title="Title", values=[["x"]] * 501)
        self.assertEqual(result["status"], "INVALID_REQUEST")
        self.assertEqual(self.requests, [])

    async def test_tokens_and_authorization_header_are_never_returned(self) -> None:
        result = await self.call(
            FakeResponse(200, {"spreadsheetId": "safe"}),
            FakeResponse(200, {}),
            title="Title",
            values=[["A"]],
        )
        serialized = json.dumps(result)
        self.assertNotIn(self.access_token, serialized)
        self.assertNotIn(self.refresh_token, serialized)
        self.assertNotIn("Authorization", serialized)


if __name__ == "__main__":
    unittest.main()