"""MCP Google Workspace semantic tools."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from time import time
from typing import Protocol
from urllib.parse import quote

import httpx
from mcp.server import MCPServer

from web import google_oauth


DRIVE_FILES_ENDPOINT = "https://www.googleapis.com/drive/v3/files"
DRIVE_FILE_FIELDS = "nextPageToken,files(id,name,mimeType,modifiedTime,createdTime,webViewLink)"
DOCS_CREATE_ENDPOINT = "https://docs.googleapis.com/v1/documents"
SHEETS_CREATE_ENDPOINT = "https://sheets.googleapis.com/v4/spreadsheets"
MAX_SHEET_ROWS = 500
MAX_SHEET_COLUMNS = 50
MAX_SHEET_CELLS = 20_000
MAX_SHEET_STRING_LENGTH = 2_000
SUPPORTED_CHART_TYPES = frozenset({"LINE", "BAR", "COLUMN", "PIE"})
A1_RANGE_PATTERN = re.compile(
    r"^(?:(?:'((?:[^']|'')+)'|([^'!]+))!)?([A-Za-z]+)([1-9][0-9]*):([A-Za-z]+)([1-9][0-9]*)$"
)


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


def _docs_error(
    status: str,
    message: str,
    document_id: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": status,
        "tool": "google_docs_create",
        "message": message,
    }
    if document_id:
        payload.update({
            "document_id": document_id,
            "url": f"https://docs.google.com/document/d/{document_id}/edit",
            "partial": True,
        })
    return payload


def _docs_http_error(status_code: int, document_id: str | None = None) -> dict[str, object]:
    if status_code == 401:
        return _docs_error("AUTH_REFRESH_FAILED", "Google authorization is no longer valid", document_id)
    if status_code == 403:
        return _docs_error("PERMISSION_DENIED", "The connected Google account cannot create this document", document_id)
    if status_code == 429:
        return _docs_error("RATE_LIMITED", "Google Docs rate limit reached; retry later", document_id)
    if status_code >= 500:
        return _docs_error("GOOGLE_API_UNAVAILABLE", "Google Docs is temporarily unavailable", document_id)
    if document_id:
        return _docs_error("DOCUMENT_CREATE_FAILED", "The document was created but its body could not be inserted", document_id)
    return _docs_error("INVALID_REQUEST", "Google Docs rejected the document request")


async def create_google_document(
    scope: GoogleToolScope,
    title: str,
    content: str,
    folder_id: str | None = None,
) -> dict[str, object]:
    normalized_title = title.strip()
    if not normalized_title or len(normalized_title) > 300:
        return _docs_error("INVALID_REQUEST", "title must contain between 1 and 300 characters")
    if not content.strip() or len(content) > 20_000:
        return _docs_error("INVALID_REQUEST", "content must contain between 1 and 20000 characters")
    if folder_id:
        return _docs_error("INVALID_REQUEST", "folder placement is not supported in this phase")
    token = scope.token_store.google_token(scope.username)
    if token is None:
        return _docs_error("NOT_CONNECTED", "Connect Google Workspace before creating Docs")
    if getattr(token, "expires_at", 0) <= int(time()) + 60:
        try:
            token = await _refresh(scope, getattr(token, "refresh_token", None))
        except google_oauth.GoogleOAuthError:
            return _docs_error("AUTH_REFRESH_FAILED", "Google authorization could not be refreshed")
        if token is None:
            return _docs_error("AUTH_REFRESH_FAILED", "Google authorization could not be refreshed")

    refreshed_after_unauthorized = False
    while True:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(
                    DOCS_CREATE_ENDPOINT,
                    json={"title": normalized_title},
                    headers={"Authorization": f"Bearer {getattr(token, 'access_token', '')}"},
                )
        except httpx.HTTPError:
            return _docs_error("GOOGLE_API_UNAVAILABLE", "Google Docs is temporarily unavailable")
        if response.status_code == 401 and not refreshed_after_unauthorized:
            try:
                token = await _refresh(scope, getattr(token, "refresh_token", None))
            except google_oauth.GoogleOAuthError:
                token = None
            if token is None:
                return _docs_error("AUTH_REFRESH_FAILED", "Google authorization could not be refreshed")
            refreshed_after_unauthorized = True
            continue
        if response.status_code >= 400:
            return _docs_http_error(response.status_code)
        try:
            create_payload = response.json()
        except ValueError:
            create_payload = None
        document_id = create_payload.get("documentId") if isinstance(create_payload, dict) else None
        if not isinstance(document_id, str) or not document_id:
            return _docs_error("DOCUMENT_CREATE_FAILED", "Google Docs did not return a document ID")
        break

    batch_endpoint = f"{DOCS_CREATE_ENDPOINT}/{document_id}:batchUpdate"
    refreshed_after_unauthorized = False
    while True:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(
                    batch_endpoint,
                    json={"requests": [{"insertText": {"location": {"index": 1}, "text": content}}]},
                    headers={"Authorization": f"Bearer {getattr(token, 'access_token', '')}"},
                )
        except httpx.HTTPError:
            return _docs_error(
                "GOOGLE_API_UNAVAILABLE", "The document was created but Google Docs is temporarily unavailable", document_id
            )
        if response.status_code == 401 and not refreshed_after_unauthorized:
            try:
                token = await _refresh(scope, getattr(token, "refresh_token", None))
            except google_oauth.GoogleOAuthError:
                token = None
            if token is None:
                return _docs_error("AUTH_REFRESH_FAILED", "Google authorization could not be refreshed", document_id)
            refreshed_after_unauthorized = True
            continue
        if response.status_code >= 400:
            return _docs_http_error(response.status_code, document_id)
        return {
            "status": "AVAILABLE",
            "tool": "google_docs_create",
            "document_id": document_id,
            "title": normalized_title,
            "url": f"https://docs.google.com/document/d/{document_id}/edit",
            "scope": google_oauth.DRIVE_FILE_SCOPE,
            "scope_limited": True,
            "content_format": "plain_text",
        }


def _sheets_error(
    status: str,
    message: str,
    spreadsheet_id: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": status,
        "tool": "google_sheets_create",
        "message": message,
    }
    if spreadsheet_id:
        payload.update({
            "spreadsheet_id": spreadsheet_id,
            "url": f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit",
            "partial": True,
        })
    return payload


def _sheets_http_error(status_code: int, spreadsheet_id: str | None = None) -> dict[str, object]:
    if status_code == 401:
        return _sheets_error("AUTH_REFRESH_FAILED", "Google authorization is no longer valid", spreadsheet_id)
    if status_code == 403:
        return _sheets_error("PERMISSION_DENIED", "The connected Google account cannot create this spreadsheet", spreadsheet_id)
    if status_code == 429:
        return _sheets_error("RATE_LIMITED", "Google Sheets rate limit reached; retry later", spreadsheet_id)
    if status_code >= 500:
        return _sheets_error("GOOGLE_API_UNAVAILABLE", "Google Sheets is temporarily unavailable", spreadsheet_id)
    if spreadsheet_id:
        return _sheets_error("VALUES_WRITE_FAILED", "The spreadsheet was created but its values could not be written", spreadsheet_id)
    return _sheets_error("INVALID_REQUEST", "Google Sheets rejected the spreadsheet request")


def _sheet_values(
    values: list[list[object]] | None,
    headers: list[object] | None,
    rows: list[list[object]] | None,
) -> list[list[object]] | None:
    if values is not None and (headers is not None or rows is not None):
        return None
    if values is None:
        if headers is None or rows is None:
            return None
        values = [headers, *rows]
    if not isinstance(values, list) or not values or len(values) > MAX_SHEET_ROWS:
        return None
    cell_count = 0
    for row in values:
        if not isinstance(row, list) or not row or len(row) > MAX_SHEET_COLUMNS:
            return None
        cell_count += len(row)
        for cell in row:
            if cell is not None and not isinstance(cell, (str, int, float, bool)):
                return None
            if isinstance(cell, str) and len(cell) > MAX_SHEET_STRING_LENGTH:
                return None
            if isinstance(cell, float) and not math.isfinite(cell):
                return None
    return values if cell_count <= MAX_SHEET_CELLS else None


async def create_google_spreadsheet(
    scope: GoogleToolScope,
    title: str,
    values: list[list[object]] | None = None,
    headers: list[object] | None = None,
    rows: list[list[object]] | None = None,
    sheet_name: str | None = None,
    start_range: str = "A1",
    folder_id: str | None = None,
) -> dict[str, object]:
    normalized_title = title.strip()
    normalized_sheet_name = sheet_name.strip() if isinstance(sheet_name, str) else None
    normalized_range = start_range.strip() if isinstance(start_range, str) else ""
    normalized_values = _sheet_values(values, headers, rows)
    if not normalized_title or len(normalized_title) > 300:
        return _sheets_error("INVALID_REQUEST", "title must contain between 1 and 300 characters")
    if normalized_values is None:
        return _sheets_error("INVALID_REQUEST", "values must be a non-empty bounded 2D array of scalar values")
    if normalized_sheet_name is not None and (not normalized_sheet_name or len(normalized_sheet_name) > 100):
        return _sheets_error("INVALID_REQUEST", "sheet_name must contain between 1 and 100 characters")
    if not normalized_range or len(normalized_range) > 100:
        return _sheets_error("INVALID_REQUEST", "start_range must contain between 1 and 100 characters")
    if folder_id:
        return _sheets_error("INVALID_REQUEST", "folder placement is not supported in this phase")
    token = scope.token_store.google_token(scope.username)
    if token is None:
        return _sheets_error("NOT_CONNECTED", "Connect Google Workspace before creating Sheets")
    if getattr(token, "expires_at", 0) <= int(time()) + 60:
        try:
            token = await _refresh(scope, getattr(token, "refresh_token", None))
        except google_oauth.GoogleOAuthError:
            return _sheets_error("AUTH_REFRESH_FAILED", "Google authorization could not be refreshed")
        if token is None:
            return _sheets_error("AUTH_REFRESH_FAILED", "Google authorization could not be refreshed")

    create_body: dict[str, object] = {"properties": {"title": normalized_title}}
    if normalized_sheet_name:
        create_body["sheets"] = [{"properties": {"title": normalized_sheet_name}}]
    refreshed_after_unauthorized = False
    while True:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(
                    SHEETS_CREATE_ENDPOINT,
                    json=create_body,
                    headers={"Authorization": f"Bearer {getattr(token, 'access_token', '')}"},
                )
        except httpx.HTTPError:
            return _sheets_error("GOOGLE_API_UNAVAILABLE", "Google Sheets is temporarily unavailable")
        if response.status_code == 401 and not refreshed_after_unauthorized:
            try:
                token = await _refresh(scope, getattr(token, "refresh_token", None))
            except google_oauth.GoogleOAuthError:
                token = None
            if token is None:
                return _sheets_error("AUTH_REFRESH_FAILED", "Google authorization could not be refreshed")
            refreshed_after_unauthorized = True
            continue
        if response.status_code >= 400:
            return _sheets_http_error(response.status_code)
        try:
            create_payload = response.json()
        except ValueError:
            create_payload = None
        spreadsheet_id = create_payload.get("spreadsheetId") if isinstance(create_payload, dict) else None
        if not isinstance(spreadsheet_id, str) or not spreadsheet_id:
            return _sheets_error("SPREADSHEET_CREATE_FAILED", "Google Sheets did not return a spreadsheet ID")
        break

    target_range = normalized_range
    if normalized_sheet_name:
        escaped_name = normalized_sheet_name.replace("'", "''")
        target_range = f"'{escaped_name}'!{normalized_range}"
    values_endpoint = f"{SHEETS_CREATE_ENDPOINT}/{spreadsheet_id}/values/{quote(target_range, safe='')}"
    refreshed_after_unauthorized = False
    while True:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.put(
                    values_endpoint,
                    params={"valueInputOption": "RAW"},
                    json={"range": target_range, "majorDimension": "ROWS", "values": normalized_values},
                    headers={"Authorization": f"Bearer {getattr(token, 'access_token', '')}"},
                )
        except httpx.HTTPError:
            return _sheets_error(
                "GOOGLE_API_UNAVAILABLE", "The spreadsheet was created but Google Sheets is temporarily unavailable", spreadsheet_id
            )
        if response.status_code == 401 and not refreshed_after_unauthorized:
            try:
                token = await _refresh(scope, getattr(token, "refresh_token", None))
            except google_oauth.GoogleOAuthError:
                token = None
            if token is None:
                return _sheets_error("AUTH_REFRESH_FAILED", "Google authorization could not be refreshed", spreadsheet_id)
            refreshed_after_unauthorized = True
            continue
        if response.status_code >= 400:
            return _sheets_http_error(response.status_code, spreadsheet_id)
        return {
            "status": "AVAILABLE",
            "tool": "google_sheets_create",
            "spreadsheet_id": spreadsheet_id,
            "title": normalized_title,
            "url": f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit",
            "rows_written": len(normalized_values),
            "columns_written": max(len(row) for row in normalized_values),
            "scope": google_oauth.DRIVE_FILE_SCOPE,
            "scope_limited": True,
        }


def _chart_error(status: str, message: str) -> dict[str, object]:
    return {"status": status, "tool": "google_sheets_add_chart", "message": message}


def _chart_http_error(status_code: int, operation: str) -> dict[str, object]:
    if status_code == 401:
        return _chart_error("AUTH_REFRESH_FAILED", "Google authorization is no longer valid")
    if status_code == 403:
        return _chart_error("PERMISSION_DENIED", "The connected Google account cannot modify this spreadsheet")
    if status_code == 404:
        return _chart_error("SPREADSHEET_NOT_FOUND", "The spreadsheet or worksheet was not found")
    if status_code == 429:
        return _chart_error("RATE_LIMITED", "Google Sheets rate limit reached; retry later")
    if status_code >= 500:
        return _chart_error("GOOGLE_API_UNAVAILABLE", "Google Sheets is temporarily unavailable")
    if operation == "chart":
        return _chart_error("CHART_CREATE_FAILED", "Google Sheets rejected the chart request")
    return _chart_error("INVALID_REQUEST", "Google Sheets rejected the spreadsheet request")


def _column_number(column: str) -> int:
    value = 0
    for character in column.upper():
        value = value * 26 + ord(character) - ord("A") + 1
    return value


def parse_a1_grid_range(data_range: str) -> tuple[str | None, dict[str, int]] | None:
    match = A1_RANGE_PATTERN.fullmatch(data_range.strip()) if isinstance(data_range, str) else None
    if match is None:
        return None
    quoted_sheet, plain_sheet, start_column, start_row, end_column, end_row = match.groups()
    sheet_name = quoted_sheet.replace("''", "'") if quoted_sheet is not None else plain_sheet
    start_column_number = _column_number(start_column)
    end_column_number = _column_number(end_column)
    start_row_number = int(start_row)
    end_row_number = int(end_row)
    if (
        end_column_number < start_column_number
        or end_row_number < start_row_number
        or end_column_number - start_column_number + 1 < 2
        or end_row_number - start_row_number + 1 < 2
        or end_column_number > 18_278
        or end_row_number > 1_000_000
    ):
        return None
    return sheet_name, {
        "startRowIndex": start_row_number - 1,
        "endRowIndex": end_row_number,
        "startColumnIndex": start_column_number - 1,
        "endColumnIndex": end_column_number,
    }


def _chart_spec(chart_type: str, grid_range: dict[str, int], title: str | None) -> dict[str, object]:
    domain_range = {**grid_range, "endColumnIndex": grid_range["startColumnIndex"] + 1}
    series_ranges = [
        {**grid_range, "startColumnIndex": column, "endColumnIndex": column + 1}
        for column in range(grid_range["startColumnIndex"] + 1, grid_range["endColumnIndex"])
    ]
    spec: dict[str, object] = {}
    if title:
        spec["title"] = title
    if chart_type == "PIE":
        spec["pieChart"] = {
            "legendPosition": "RIGHT_LEGEND",
            "domain": {"sourceRange": {"sources": [domain_range]}},
            "series": {"sourceRange": {"sources": [series_ranges[0]]}},
            "threeDimensional": False,
        }
    else:
        spec["basicChart"] = {
            "chartType": chart_type,
            "legendPosition": "BOTTOM_LEGEND",
            "axis": [
                {"position": "BOTTOM_AXIS", "title": ""},
                {"position": "LEFT_AXIS", "title": ""},
            ],
            "domains": [{"domain": {"sourceRange": {"sources": [domain_range]}}}],
            "series": [
                {"series": {"sourceRange": {"sources": [series_range]}}}
                for series_range in series_ranges
            ],
            "headerCount": 1,
        }
    return spec


async def add_google_sheets_chart(
    scope: GoogleToolScope,
    spreadsheet_id: str,
    chart_type: str,
    data_range: str,
    title: str | None = None,
    sheet_id: int | None = None,
    anchor_row: int | None = None,
    anchor_column: int | None = None,
) -> dict[str, object]:
    normalized_spreadsheet_id = spreadsheet_id.strip() if isinstance(spreadsheet_id, str) else ""
    normalized_chart_type = chart_type.strip().upper() if isinstance(chart_type, str) else ""
    normalized_title = title.strip() if isinstance(title, str) else None
    parsed_range = parse_a1_grid_range(data_range)
    if not normalized_spreadsheet_id or len(normalized_spreadsheet_id) > 200 or not re.fullmatch(r"[A-Za-z0-9_-]+", normalized_spreadsheet_id):
        return _chart_error("INVALID_REQUEST", "spreadsheet_id is invalid")
    if normalized_chart_type not in SUPPORTED_CHART_TYPES:
        return _chart_error("CHART_TYPE_UNSUPPORTED", "chart_type must be LINE, BAR, COLUMN, or PIE")
    if parsed_range is None:
        return _chart_error("RANGE_INVALID", "data_range must be a rectangular A1 range with headers and data")
    if normalized_title is not None and (not normalized_title or len(normalized_title) > 300):
        return _chart_error("INVALID_REQUEST", "title must contain between 1 and 300 characters")
    for name, value in (("sheet_id", sheet_id), ("anchor_row", anchor_row), ("anchor_column", anchor_column)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            return _chart_error("INVALID_REQUEST", f"{name} must be a non-negative integer")
    sheet_name, grid_range = parsed_range
    if normalized_chart_type == "PIE" and grid_range["endColumnIndex"] - grid_range["startColumnIndex"] != 2:
        return _chart_error("RANGE_INVALID", "PIE charts require exactly one domain and one series column")

    token = scope.token_store.google_token(scope.username)
    if token is None:
        return _chart_error("NOT_CONNECTED", "Connect Google Workspace before adding charts")
    if getattr(token, "expires_at", 0) <= int(time()) + 60:
        try:
            token = await _refresh(scope, getattr(token, "refresh_token", None))
        except google_oauth.GoogleOAuthError:
            return _chart_error("AUTH_REFRESH_FAILED", "Google authorization could not be refreshed")
        if token is None:
            return _chart_error("AUTH_REFRESH_FAILED", "Google authorization could not be refreshed")

    if sheet_id is None:
        metadata_endpoint = f"{SHEETS_CREATE_ENDPOINT}/{normalized_spreadsheet_id}"
        refreshed_after_unauthorized = False
        while True:
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    response = await client.get(
                        metadata_endpoint,
                        params={"fields": "sheets.properties(sheetId,title)"},
                        headers={"Authorization": f"Bearer {getattr(token, 'access_token', '')}"},
                    )
            except httpx.HTTPError:
                return _chart_error("GOOGLE_API_UNAVAILABLE", "Google Sheets is temporarily unavailable")
            if response.status_code == 401 and not refreshed_after_unauthorized:
                try:
                    token = await _refresh(scope, getattr(token, "refresh_token", None))
                except google_oauth.GoogleOAuthError:
                    token = None
                if token is None:
                    return _chart_error("AUTH_REFRESH_FAILED", "Google authorization could not be refreshed")
                refreshed_after_unauthorized = True
                continue
            if response.status_code >= 400:
                return _chart_http_error(response.status_code, "metadata")
            try:
                metadata = response.json()
            except ValueError:
                metadata = None
            sheets = metadata.get("sheets") if isinstance(metadata, dict) else None
            if not isinstance(sheets, list):
                return _chart_error("SPREADSHEET_NOT_FOUND", "The spreadsheet has no accessible worksheets")
            properties = [item.get("properties") for item in sheets if isinstance(item, dict)]
            target = next((item for item in properties if isinstance(item, dict) and (
                sheet_name is None or item.get("title") == sheet_name
            )), None)
            resolved_sheet_id = target.get("sheetId") if isinstance(target, dict) else None
            if isinstance(resolved_sheet_id, bool) or not isinstance(resolved_sheet_id, int):
                return _chart_error("SPREADSHEET_NOT_FOUND", "The requested worksheet was not found")
            sheet_id = resolved_sheet_id
            break

    source_range = {"sheetId": sheet_id, **grid_range}
    resolved_anchor_row = anchor_row if anchor_row is not None else grid_range["startRowIndex"]
    resolved_anchor_column = anchor_column if anchor_column is not None else grid_range["endColumnIndex"] + 1
    chart = {
        "spec": _chart_spec(normalized_chart_type, source_range, normalized_title),
        "position": {
            "overlayPosition": {
                "anchorCell": {
                    "sheetId": sheet_id,
                    "rowIndex": resolved_anchor_row,
                    "columnIndex": resolved_anchor_column,
                },
                "offsetXPixels": 0,
                "offsetYPixels": 0,
            }
        },
    }
    batch_endpoint = f"{SHEETS_CREATE_ENDPOINT}/{normalized_spreadsheet_id}:batchUpdate"
    refreshed_after_unauthorized = False
    while True:
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(
                    batch_endpoint,
                    json={"requests": [{"addChart": {"chart": chart}}]},
                    headers={"Authorization": f"Bearer {getattr(token, 'access_token', '')}"},
                )
        except httpx.HTTPError:
            return _chart_error("GOOGLE_API_UNAVAILABLE", "Google Sheets is temporarily unavailable")
        if response.status_code == 401 and not refreshed_after_unauthorized:
            try:
                token = await _refresh(scope, getattr(token, "refresh_token", None))
            except google_oauth.GoogleOAuthError:
                token = None
            if token is None:
                return _chart_error("AUTH_REFRESH_FAILED", "Google authorization could not be refreshed")
            refreshed_after_unauthorized = True
            continue
        if response.status_code >= 400:
            return _chart_http_error(response.status_code, "chart")
        try:
            payload = response.json()
        except ValueError:
            payload = None
        replies = payload.get("replies") if isinstance(payload, dict) else None
        add_chart = replies[0].get("addChart") if isinstance(replies, list) and replies and isinstance(replies[0], dict) else None
        returned_chart = add_chart.get("chart") if isinstance(add_chart, dict) else None
        chart_id = returned_chart.get("chartId") if isinstance(returned_chart, dict) else None
        if isinstance(chart_id, bool) or not isinstance(chart_id, int):
            return _chart_error("CHART_CREATE_FAILED", "Google Sheets did not return a chart ID")
        return {
            "status": "AVAILABLE",
            "tool": "google_sheets_add_chart",
            "spreadsheet_id": normalized_spreadsheet_id,
            "chart_id": chart_id,
            "chart_type": normalized_chart_type,
            "title": normalized_title,
            "data_range": data_range.strip(),
            "sheet_id": sheet_id,
            "url": f"https://docs.google.com/spreadsheets/d/{normalized_spreadsheet_id}/edit",
            "scope": google_oauth.DRIVE_FILE_SCOPE,
            "scope_limited": True,
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

    @server.tool(
        name="google_docs_create",
        description=(
            "Create a Google Docs document for the connected user and insert the supplied content as plain text. "
            "Returns the document ID and Google Docs URL."
        ),
        structured_output=True,
    )
    async def scoped_google_docs_create(
        title: str,
        content: str,
        folder_id: str | None = None,
    ) -> dict[str, object]:
        return await create_google_document(scope, title, content, folder_id)

    @server.tool(
        name="google_sheets_create",
        description=(
            "Create a Google Sheets spreadsheet for the connected user and write a bounded 2D scalar values array. "
            "Legacy headers and rows inputs remain supported."
        ),
        structured_output=True,
    )
    async def scoped_google_sheets_create(
        title: str,
        values: list[list[object]] | None = None,
        headers: list[object] | None = None,
        rows: list[list[object]] | None = None,
        sheet_name: str | None = None,
        start_range: str = "A1",
        folder_id: str | None = None,
    ) -> dict[str, object]:
        return await create_google_spreadsheet(
            scope, title, values, headers, rows, sheet_name, start_range, folder_id
        )

    @server.tool(
        name="google_sheets_add_chart",
        description="Add a LINE, BAR, COLUMN, or PIE embedded chart using a bounded A1 data range.",
        structured_output=True,
    )
    async def scoped_google_sheets_add_chart(
        spreadsheet_id: str,
        chart_type: str,
        data_range: str,
        title: str | None = None,
        sheet_id: int | None = None,
        anchor_row: int | None = None,
        anchor_column: int | None = None,
    ) -> dict[str, object]:
        return await add_google_sheets_chart(
            scope, spreadsheet_id, chart_type, data_range, title, sheet_id, anchor_row, anchor_column
        )

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


@GOOGLE_MCP.tool(
    description="Add a supported embedded chart to an existing Google Sheet using a validated A1 range.",
    structured_output=True,
)
def google_sheets_add_chart(
    spreadsheet_id: str,
    chart_type: str,
    data_range: str,
    title: str | None = None,
    sheet_id: int | None = None,
    anchor_row: int | None = None,
    anchor_column: int | None = None,
) -> dict[str, object]:
    return _unconfigured(
        "google_sheets_add_chart",
        spreadsheet_id=spreadsheet_id,
        chart_type=chart_type,
        data_range=data_range,
    )


if __name__ == "__main__":
    GOOGLE_MCP.run(transport="stdio")
