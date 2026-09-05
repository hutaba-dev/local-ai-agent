"""MCP Google Workspace semantic tools."""

from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Protocol

import httpx
from mcp.server import MCPServer

from web import google_oauth


DRIVE_FILES_ENDPOINT = "https://www.googleapis.com/drive/v3/files"
DRIVE_FILE_FIELDS = "nextPageToken,files(id,name,mimeType,modifiedTime,createdTime,webViewLink)"


class GoogleTokenStore(Protocol):
    def google_token(self, username: str) -> object | None: ...

    def save_google_token(
        self,
        username: str,
        access_token: str,
        refresh_token: str | None,
        expires_at: int,
        scopes: tuple[str, ...],
        token_type: str,
    ) -> None: ...


@dataclass(frozen=True)
class GoogleToolScope:
    username: str
    token_store: GoogleTokenStore

GOOGLE_MCP = MCPServer(
    "ahnbys-google",
    description="Semantic Google Workspace document operations (Drive, Docs, Sheets) for authorized workspace access.",
    version="1.0.0",
)

_UNCONFIGURED_REASON = (
    "Google Workspace is not connected yet: credentials and the Google API "
    "implementation are not available in this phase. No document, file, or "
    "sheet was read or created."
)


def _unconfigured(tool: str, **details: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "UNCONFIGURED",
        "tool": tool,
        "reason": _UNCONFIGURED_REASON,
    }
    payload.update(details)
    return payload


@GOOGLE_MCP.tool(
    description=(
        "List or search Google Drive files the user can access. "
        "Use this when the user asks to find, list, or review files in their Google Drive. "
        "Returns bounded file metadata only; never returns file contents."
    ),
    structured_output=True,
)
def google_drive_list(
    query: str | None = None,
    mime_type: str | None = None,
    limit: int = 20,
    page_size: int | None = None,
    page_token: str | None = None,
) -> dict[str, object]:
    return _unconfigured("google_drive_list", query=query, mime_type=mime_type, limit=limit, files=[])


def _safe_error(status: str, message: str) -> dict[str, object]:
    return {"status": status, "tool": "google_drive_list", "message": message, "files": []}


def _drive_query(query: str | None, mime_type: str | None) -> str:
    clauses = ["trashed = false"]
    if query and query.strip():
        escaped = query.strip().replace("\\", "\\\\").replace("'", "\\'")
        clauses.append(f"name contains '{escaped}'")
    if mime_type and mime_type.strip():
        escaped = mime_type.strip().replace("\\", "\\\\").replace("'", "\\'")
        clauses.append(f"mimeType = '{escaped}'")
    return " and ".join(clauses)


def _normalize_files(payload: object) -> tuple[list[dict[str, object]], str | None] | None:
    if not isinstance(payload, dict) or not isinstance(payload.get("files", []), list):
        return None
    allowed = ("id", "name", "mimeType", "modifiedTime", "createdTime", "webViewLink")
    files = [
        {key: item[key] for key in allowed if key in item}
        for item in payload.get("files", [])
        if isinstance(item, dict)
    ]
    next_page_token = payload.get("nextPageToken")
    return files, next_page_token if isinstance(next_page_token, str) else None


async def _refresh(scope: GoogleToolScope, refresh_token: str | None) -> object | None:
    if not refresh_token:
        return None
    refreshed = await google_oauth.refresh_access_token(refresh_token)
    scope.token_store.save_google_token(
        scope.username,
        refreshed.access_token,
        refreshed.refresh_token,
        refreshed.expires_at,
        refreshed.scopes,
        refreshed.token_type,
    )
    return scope.token_store.google_token(scope.username)


async def list_drive_files(
    scope: GoogleToolScope,
    query: str | None = None,
    mime_type: str | None = None,
    limit: int = 20,
    page_size: int | None = None,
    page_token: str | None = None,
) -> dict[str, object]:
    effective_page_size = limit if page_size is None else page_size
    if not 1 <= effective_page_size <= 100 or not 1 <= limit <= 100:
        return _safe_error("INVALID_REQUEST", "limit and page_size must be between 1 and 100")
    if page_token is not None and (not page_token.strip() or len(page_token) > 2048):
        return _safe_error("INVALID_REQUEST", "page_token is invalid")
    token = scope.token_store.google_token(scope.username)
    if token is None:
        return _safe_error("NOT_CONNECTED", "Connect Google Workspace before using Drive tools")
    if getattr(token, "expires_at", 0) <= int(time()) + 60:
        try:
            token = await _refresh(scope, getattr(token, "refresh_token", None))
        except google_oauth.GoogleOAuthError:
            return _safe_error("AUTH_REFRESH_FAILED", "Google authorization could not be refreshed")
        if token is None:
            return _safe_error("AUTH_REFRESH_FAILED", "Google authorization could not be refreshed")

    params: dict[str, object] = {
        "pageSize": effective_page_size,
        "fields": DRIVE_FILE_FIELDS,
        "q": _drive_query(query, mime_type),
        "spaces": "drive",
    }
    if page_token:
        params["pageToken"] = page_token

    refreshed_after_unauthorized = False
    while True:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.get(
                    DRIVE_FILES_ENDPOINT,
                    params=params,
                    headers={"Authorization": f"Bearer {getattr(token, 'access_token', '')}"},
                )
        except httpx.HTTPError:
            return _safe_error("GOOGLE_API_UNAVAILABLE", "Google Drive is temporarily unavailable")
        if response.status_code == 401 and not refreshed_after_unauthorized:
            try:
                token = await _refresh(scope, getattr(token, "refresh_token", None))
            except google_oauth.GoogleOAuthError:
                token = None
            if token is None:
                return _safe_error("AUTH_REFRESH_FAILED", "Google authorization could not be refreshed")
            refreshed_after_unauthorized = True
            continue
        if response.status_code == 401:
            return _safe_error("AUTH_REFRESH_FAILED", "Google authorization is no longer valid")
        if response.status_code == 403:
            return _safe_error("PERMISSION_DENIED", "The connected Google account did not grant access to these files")
        if response.status_code == 429:
            return _safe_error("RATE_LIMITED", "Google Drive rate limit reached; retry later")
        if response.status_code >= 500:
            return _safe_error("GOOGLE_API_UNAVAILABLE", "Google Drive is temporarily unavailable")
        if response.status_code >= 400:
            return _safe_error("INVALID_REQUEST", "Google Drive rejected the request")
        try:
            normalized = _normalize_files(response.json())
        except ValueError:
            normalized = None
        if normalized is None:
            return _safe_error("GOOGLE_API_UNAVAILABLE", "Google Drive returned an invalid response")
        files, next_page_token = normalized
        return {
            "status": "AVAILABLE",
            "tool": "google_drive_list",
            "scope": google_oauth.DRIVE_FILE_SCOPE,
            "scope_limited": True,
            "files": files,
            "next_page_token": next_page_token,
        }


def create_google_mcp(scope: GoogleToolScope) -> MCPServer:
    server = MCPServer(
        "ahnbys-google-scoped",
        description="User-scoped Google Workspace document operations.",
        version="1.1.0",
    )

    @server.tool(
        name="google_drive_list",
        description=(
            "List or search only Google Drive files available through the connected user's drive.file grant. "
            "Returns bounded metadata and never file contents."
        ),
        structured_output=True,
    )
    async def scoped_google_drive_list(
        query: str | None = None,
        mime_type: str | None = None,
        limit: int = 20,
        page_size: int | None = None,
        page_token: str | None = None,
    ) -> dict[str, object]:
        return await list_drive_files(scope, query, mime_type, limit, page_size, page_token)

    @server.tool(name="google_docs_create", description="Google Docs creation is not configured in this phase.", structured_output=True)
    def scoped_google_docs_create(title: str, content: str, folder_id: str | None = None) -> dict[str, object]:
        return google_docs_create(title, content, folder_id)

    @server.tool(name="google_sheets_create", description="Google Sheets creation is not configured in this phase.", structured_output=True)
    def scoped_google_sheets_create(
        title: str,
        headers: list[str],
        rows: list[list[str]],
        sheet_name: str | None = None,
        folder_id: str | None = None,
    ) -> dict[str, object]:
        return google_sheets_create(title, headers, rows, sheet_name, folder_id)

    return server


@GOOGLE_MCP.tool(
    description=(
        "Create a Google Docs document from a user-provided title and content. "
        "Use this when the user asks to turn a report, notes, or structured text into a Google Doc. "
        "Content is plain text or Markdown, never a Google Docs batchUpdate payload."
    ),
    structured_output=True,
)
def google_docs_create(
    title: str,
    content: str,
    folder_id: str | None = None,
) -> dict[str, object]:
    title = title.strip()
    if not title or len(title) > 300:
        raise ValueError("title must contain between 1 and 300 characters")
    if not content.strip():
        raise ValueError("content must not be empty")
    return _unconfigured(
        "google_docs_create",
        title=title,
        folder_id=folder_id,
        document_url=None,
    )


@GOOGLE_MCP.tool(
    description=(
        "Create a Google Sheets spreadsheet from tabular data. "
        "Use this when the user asks to turn a table, comparison, or structured dataset into a Google Sheet. "
        "Provide column headers and rows as plain values, never a Google Sheets API request body."
    ),
    structured_output=True,
)
def google_sheets_create(
    title: str,
    headers: list[str],
    rows: list[list[str]],
    sheet_name: str | None = None,
    folder_id: str | None = None,
) -> dict[str, object]:
    title = title.strip()
    if not title or len(title) > 300:
        raise ValueError("title must contain between 1 and 300 characters")
    if not headers:
        raise ValueError("headers must not be empty")
    return _unconfigured(
        "google_sheets_create",
        title=title,
        sheet_name=sheet_name,
        folder_id=folder_id,
        spreadsheet_url=None,
    )


if __name__ == "__main__":
    GOOGLE_MCP.run(transport="stdio")
