from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from mcp_servers.google_server import DOCS_CREATE_ENDPOINT, GoogleToolScope, create_google_document
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

    async def get(self, url: str, **kwargs: object) -> FakeResponse:
        self.requests.append({"method": "GET", "url": url, **kwargs})
        return self.responses.pop(0)


class GoogleDocsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.access_token = "docs-access-token-never-return"
        self.refresh_token = "docs-refresh-token-never-return"
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
            return await create_google_document(self.scope, **arguments)

    async def test_connected_user_creates_document_and_inserts_content(self) -> None:
        result = await self.call(
            FakeResponse(200, {"documentId": "doc-123", "title": "Ignored provider title"}),
            FakeResponse(200, {"replies": [{}]}),
            title="AHNBYS report",
            content="Report body",
        )

        self.assertEqual(result, {
            "status": "AVAILABLE",
            "tool": "google_docs_create",
            "document_id": "doc-123",
            "title": "AHNBYS report",
            "url": "https://docs.google.com/document/d/doc-123/edit",
            "scope": DRIVE_FILE_SCOPE,
            "scope_limited": True,
            "content_format": "native_google_docs",
            "heading_count": 0,
            "table_count": 0,
        })
        self.assertEqual(self.requests[0]["url"], DOCS_CREATE_ENDPOINT)
        self.assertEqual(self.requests[0]["json"], {"title": "AHNBYS report"})
        requests = self.requests[1]["json"]["requests"]
        self.assertEqual(requests[0], {"insertText": {"location": {"index": 1}, "text": "Report body\n"}})
        self.assertEqual(requests[1]["updateParagraphStyle"]["paragraphStyle"]["namedStyleType"], "NORMAL_TEXT")

    async def test_markdown_creates_native_headings_bold_and_table(self) -> None:
        document_structure = {"body": {"content": [{
            "startIndex": 15,
            "table": {"tableRows": [
                {"tableCells": [{"content": [{"startIndex": 17}]}, {"content": [{"startIndex": 20}]}]},
                {"tableCells": [{"content": [{"startIndex": 24}]}, {"content": [{"startIndex": 27}]}]},
            ]},
        }]}}
        result = await self.call(
            FakeResponse(200, {"documentId": "doc-native"}),
            FakeResponse(200, {}),
            FakeResponse(200, document_structure),
            FakeResponse(200, {}),
            title="Native report",
            content=(
                "# Report\n## Findings\nA **verified** fact.\n"
                "| Supplier | Capacity |\n|---|---:|\n| Alpha | 100000 |"
            ),
        )

        self.assertEqual(result["status"], "AVAILABLE")
        self.assertEqual(result["content_format"], "native_google_docs")
        self.assertEqual(result["heading_count"], 2)
        self.assertEqual(result["table_count"], 1)
        initial_requests = self.requests[1]["json"]["requests"]
        inserted_text = initial_requests[0]["insertText"]["text"]
        self.assertNotIn("#", inserted_text)
        self.assertNotIn("**", inserted_text)
        self.assertNotIn("|---", inserted_text)
        styles = [item["updateParagraphStyle"]["paragraphStyle"]["namedStyleType"] for item in initial_requests if "updateParagraphStyle" in item]
        self.assertEqual(styles[:2], ["HEADING_1", "HEADING_2"])
        self.assertTrue(any("updateTextStyle" in item for item in initial_requests))
        self.assertTrue(any("insertTable" in item for item in initial_requests))
        self.assertEqual(self.requests[2]["method"], "GET")
        table_requests = self.requests[3]["json"]["requests"]
        table_values = [item["insertText"]["text"] for item in table_requests if "insertText" in item]
        self.assertCountEqual(table_values, ["Supplier", "Capacity", "Alpha", "100000"])
        header_styles = [item for item in table_requests if "updateTextStyle" in item]
        self.assertEqual(len(header_styles), 2)
        self.assertTrue(all(item["updateTextStyle"]["textStyle"]["bold"] for item in header_styles))

    async def test_disconnected_user_returns_not_connected(self) -> None:
        self.store.token = None

        result = await self.call(title="Title", content="Body")

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
                FakeResponse(200, {"documentId": "doc-refresh"}),
                FakeResponse(200, {}),
                title="Title",
                content="Body",
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
            result = await self.call(title="Title", content="Body")

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
                FakeResponse(200, {"documentId": "doc-after-refresh"}),
                FakeResponse(200, {}),
                title="Title",
                content="Body",
            )

        self.assertEqual(result["status"], "AVAILABLE")
        self.assertEqual(len(self.requests), 3)
        refresh.assert_awaited_once_with(self.refresh_token)

    async def test_create_403_is_permission_denied(self) -> None:
        result = await self.call(FakeResponse(403, {"error": "private"}), title="Title", content="Body")
        self.assertEqual(result["status"], "PERMISSION_DENIED")
        self.assertNotIn("private", json.dumps(result))

    async def test_create_429_is_rate_limited(self) -> None:
        result = await self.call(FakeResponse(429, {}), title="Title", content="Body")
        self.assertEqual(result["status"], "RATE_LIMITED")

    async def test_create_5xx_is_unavailable(self) -> None:
        result = await self.call(FakeResponse(503, {}), title="Title", content="Body")
        self.assertEqual(result["status"], "GOOGLE_API_UNAVAILABLE")

    async def test_batch_update_failure_reports_partial_document_safely(self) -> None:
        result = await self.call(
            FakeResponse(200, {"documentId": "doc-partial"}),
            FakeResponse(400, {"error": {"message": "private body"}}),
            title="Title",
            content="Body",
        )

        self.assertEqual(result["status"], "DOCUMENT_CREATE_FAILED")
        self.assertEqual(result["document_id"], "doc-partial")
        self.assertTrue(result["partial"])
        self.assertNotIn("private body", json.dumps(result))

    async def test_empty_title_and_content_are_invalid_without_http(self) -> None:
        empty_title = await self.call(title=" ", content="Body")
        empty_content = await self.call(title="Title", content=" ")

        self.assertEqual(empty_title["status"], "INVALID_REQUEST")
        self.assertEqual(empty_content["status"], "INVALID_REQUEST")
        self.assertEqual(self.requests, [])

    async def test_tokens_and_authorization_header_are_never_returned(self) -> None:
        result = await self.call(
            FakeResponse(200, {"documentId": "doc-safe"}),
            FakeResponse(200, {}),
            title="Title",
            content="Body",
        )
        serialized = json.dumps(result)

        self.assertNotIn(self.access_token, serialized)
        self.assertNotIn(self.refresh_token, serialized)
        self.assertNotIn("Authorization", serialized)


if __name__ == "__main__":
    unittest.main()