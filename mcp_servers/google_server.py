"""MCP Google Workspace semantic tool skeleton (Phase 2).

This module intentionally exposes only three semantic tools — Drive listing,
Docs creation, and Sheets creation — using the existing AHNBYS MCP server
pattern. It imports no Google SDK, performs no OAuth, and makes no real
Google API calls. Every tool returns the AHNBYS UNCONFIGURED state contract
until a credential-backed implementation lands in a later phase. No fake
document URLs, file IDs, or sheet IDs are ever produced.
"""

from __future__ import annotations

from mcp.server import MCPServer

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
) -> dict[str, object]:
    if limit < 1 or limit > 100:
        raise ValueError("limit must be between 1 and 100")
    return _unconfigured(
        "google_drive_list",
        query=query,
        mime_type=mime_type,
        limit=limit,
        files=[],
    )


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
