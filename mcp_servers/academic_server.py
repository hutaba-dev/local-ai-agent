"""High-level MCP facade for existing AHNBYS Academic Intelligence."""

from __future__ import annotations

from mcp.server import MCPServer

from runtime.academic_intelligence import academic_intelligence
from runtime.web_search import academic_papers


ACADEMIC_MCP = MCPServer(
    "ahnbys-academic",
    description="Researcher identity and publication evidence from existing AHNBYS Academic Intelligence.",
    version="1.0.0",
)


def _query(value: str) -> str:
    value = value.strip()
    if not value or len(value) > 500:
        raise ValueError("query must contain between 1 and 500 characters")
    return value


@ACADEMIC_MCP.tool(
    description="Resolve a researcher identity and retrieve a normalized profile with provider-specific publication and citation provenance.",
    structured_output=True,
)
def researcher_profile(query: str) -> dict[str, object]:
    result = academic_intelligence(_query(query))
    return {"status": "AVAILABLE", "entity": result}


@ACADEMIC_MCP.tool(
    description="Search normalized scholarly publication metadata when academic evidence materially improves the answer.",
    structured_output=True,
)
def publication_search(query: str, limit: int = 5) -> dict[str, object]:
    if not 1 <= limit <= 10:
        raise ValueError("limit must be between 1 and 10")
    diagnostics: list[dict[str, object]] = []
    publications = academic_papers((_query(query),), limit_per_query=limit, diagnostics=diagnostics)
    return {
        "status": "AVAILABLE" if publications else "DEGRADED",
        "publications": publications[:limit],
        "provenance": diagnostics,
    }


if __name__ == "__main__":
    ACADEMIC_MCP.run(transport="stdio")
