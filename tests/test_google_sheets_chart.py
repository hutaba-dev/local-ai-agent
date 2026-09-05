from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from mcp_servers.google_server import (
    SHEETS_CREATE_ENDPOINT,
    GoogleToolScope,
    add_google_sheets_chart,
    parse_a1_grid_range,
)
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

    async def get(self, url: str, **kwargs: object) -> FakeResponse:
        self.requests.append({"method": "GET", "url": url, **kwargs})
        return self.responses.pop(0)

    async def post(self, url: str, **kwargs: object) -> FakeResponse:
        self.requests.append({"method": "POST", "url": url, **kwargs})
        return self.responses.pop(0)


def chart_response(chart_id: int = 77) -> FakeResponse:
    return FakeResponse(200, {"replies": [{"addChart": {"chart": {"chartId": chart_id}}}]})


class GoogleSheetsChartTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.access_token = "chart-access-token-never-return"
        self.refresh_token = "chart-refresh-token-never-return"
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
            return await add_google_sheets_chart(self.scope, **arguments)

    async def test_supported_chart_types_create_expected_specs(self) -> None:
        for chart_type in ("BAR", "LINE", "COLUMN", "PIE"):
            with self.subTest(chart_type=chart_type):
                self.requests.clear()
                result = await self.call(
                    chart_response(),
                    spreadsheet_id="spreadsheet-1",
                    chart_type=chart_type,
                    data_range="A1:B5",
                    title="Integration chart",
                    sheet_id=0,
                )
                spec = self.requests[0]["json"]["requests"][0]["addChart"]["chart"]["spec"]
                self.assertEqual(result["status"], "AVAILABLE")
                self.assertEqual(result["chart_id"], 77)
                self.assertEqual(result["chart_type"], chart_type)
                self.assertEqual(result["title"], "Integration chart")
                if chart_type == "PIE":
                    self.assertIn("pieChart", spec)
                else:
                    self.assertEqual(spec["basicChart"]["chartType"], chart_type)

    def test_a1_range_converts_to_zero_based_exclusive_grid_range(self) -> None:
        self.assertEqual(parse_a1_grid_range("A1:B5"), (None, {
            "startRowIndex": 0,
            "endRowIndex": 5,
            "startColumnIndex": 0,
            "endColumnIndex": 2,
        }))

    async def test_named_sheet_is_resolved_with_minimal_metadata(self) -> None:
        result = await self.call(
            FakeResponse(200, {"sheets": [
                {"properties": {"sheetId": 7, "title": "Other"}},
                {"properties": {"sheetId": 42, "title": "Sheet1"}},
            ]}),
            chart_response(88),
            spreadsheet_id="spreadsheet-1",
            chart_type="BAR",
            data_range="Sheet1!A1:B5",
        )

        self.assertEqual(result["sheet_id"], 42)
        self.assertEqual(self.requests[0]["method"], "GET")
        self.assertEqual(self.requests[0]["params"], {"fields": "sheets.properties(sheetId,title)"})
        source = self.requests[1]["json"]["requests"][0]["addChart"]["chart"]["spec"]["basicChart"]
        self.assertEqual(source["domains"][0]["domain"]["sourceRange"]["sources"][0], {
            "sheetId": 42,
            "startRowIndex": 0,
            "endRowIndex": 5,
            "startColumnIndex": 0,
            "endColumnIndex": 1,
        })
        self.assertEqual(source["series"][0]["series"]["sourceRange"]["sources"][0]["startColumnIndex"], 1)
        anchor = self.requests[1]["json"]["requests"][0]["addChart"]["chart"]["position"]["overlayPosition"]["anchorCell"]
        self.assertEqual(anchor, {"sheetId": 42, "rowIndex": 0, "columnIndex": 3})

    async def test_invalid_ranges_do_not_call_google(self) -> None:
        for data_range in ("A1", "B5:A1", "A0:B5", "A1:A5", "Sheet1!bad"):
            with self.subTest(data_range=data_range):
                result = await self.call(
                    spreadsheet_id="spreadsheet-1", chart_type="BAR", data_range=data_range, sheet_id=0
                )
                self.assertEqual(result["status"], "RANGE_INVALID")
        self.assertEqual(self.requests, [])

    async def test_unsupported_chart_type_does_not_call_google(self) -> None:
        result = await self.call(
            spreadsheet_id="spreadsheet-1", chart_type="SCATTER", data_range="A1:B5", sheet_id=0
        )
        self.assertEqual(result["status"], "CHART_TYPE_UNSUPPORTED")
        self.assertEqual(self.requests, [])

    async def test_missing_spreadsheet_is_normalized(self) -> None:
        result = await self.call(
            FakeResponse(404, {"error": "private"}),
            spreadsheet_id="missing", chart_type="BAR", data_range="A1:B5",
        )
        self.assertEqual(result["status"], "SPREADSHEET_NOT_FOUND")
        self.assertNotIn("private", json.dumps(result))

    async def test_disconnected_user_does_not_call_google(self) -> None:
        self.store.token = None
        result = await self.call(
            spreadsheet_id="spreadsheet-1", chart_type="BAR", data_range="A1:B5", sheet_id=0
        )
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
                chart_response(), spreadsheet_id="spreadsheet-1", chart_type="BAR", data_range="A1:B5", sheet_id=0
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
            result = await self.call(
                spreadsheet_id="spreadsheet-1", chart_type="BAR", data_range="A1:B5", sheet_id=0
            )
        self.assertEqual(result["status"], "AUTH_REFRESH_FAILED")
        self.assertNotIn("provider detail", json.dumps(result))

    async def test_401_refreshes_once_then_succeeds(self) -> None:
        refreshed = OAuthTokenResponse("new-access", None, 4_000_000_000, (DRIVE_FILE_SCOPE,), "Bearer")
        with patch(
            "mcp_servers.google_server.google_oauth.refresh_access_token",
            AsyncMock(return_value=refreshed),
        ) as refresh:
            result = await self.call(
                FakeResponse(401, {}), chart_response(),
                spreadsheet_id="spreadsheet-1", chart_type="BAR", data_range="A1:B5", sheet_id=0,
            )
        self.assertEqual(result["status"], "AVAILABLE")
        self.assertEqual(len(self.requests), 2)
        refresh.assert_awaited_once_with(self.refresh_token)

    async def test_403_is_permission_denied(self) -> None:
        result = await self.call(
            FakeResponse(403, {}), spreadsheet_id="spreadsheet-1", chart_type="BAR", data_range="A1:B5", sheet_id=0
        )
        self.assertEqual(result["status"], "PERMISSION_DENIED")

    async def test_429_is_rate_limited(self) -> None:
        result = await self.call(
            FakeResponse(429, {}), spreadsheet_id="spreadsheet-1", chart_type="BAR", data_range="A1:B5", sheet_id=0
        )
        self.assertEqual(result["status"], "RATE_LIMITED")

    async def test_5xx_is_unavailable(self) -> None:
        result = await self.call(
            FakeResponse(503, {}), spreadsheet_id="spreadsheet-1", chart_type="BAR", data_range="A1:B5", sheet_id=0
        )
        self.assertEqual(result["status"], "GOOGLE_API_UNAVAILABLE")

    async def test_missing_chart_id_is_create_failed(self) -> None:
        result = await self.call(
            FakeResponse(200, {"replies": [{}]}),
            spreadsheet_id="spreadsheet-1", chart_type="BAR", data_range="A1:B5", sheet_id=0,
        )
        self.assertEqual(result["status"], "CHART_CREATE_FAILED")

    async def test_tokens_and_authorization_header_are_never_returned(self) -> None:
        result = await self.call(
            chart_response(), spreadsheet_id="spreadsheet-1", chart_type="BAR", data_range="A1:B5", sheet_id=0
        )
        serialized = json.dumps(result)
        self.assertNotIn(self.access_token, serialized)
        self.assertNotIn(self.refresh_token, serialized)
        self.assertNotIn("Authorization", serialized)


if __name__ == "__main__":
    unittest.main()