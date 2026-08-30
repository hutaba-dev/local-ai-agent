"""Bounded semantic MCP facade for existing AHNBYS Academic Intelligence."""

from __future__ import annotations

import json

from mcp.server import MCPServer

from runtime.academic_intelligence import academic_intelligence
from runtime.web_search import academic_papers


MAX_ACADEMIC_OUTPUT_CHARS = 12_000


ACADEMIC_MCP = MCPServer(
    "ahnbys-academic",
    description="Read-only scholarly identity, publication, citation, and source-coverage evidence.",
    version="1.1.0",
)


def _query(value: str) -> str:
    value = value.strip()
    if not value or len(value) > 500:
        raise ValueError("query must contain between 1 and 500 characters")
    return value


def _status(result: dict[str, object], *, require_identity: bool = False) -> str:
    states = result.get("source_status", {})
    values = set(states.values()) if isinstance(states, dict) else set()
    if not values & {"AVAILABLE_FULL", "AVAILABLE_LIMITED"}:
        return "UNAVAILABLE"
    researcher = result.get("researcher", {})
    confidence = researcher.get("identity_confidence") if isinstance(researcher, dict) else None
    if require_identity and confidence in {"LOW", "AMBIGUOUS", "UNRESOLVED", None}:
        return "DEGRADED"
    return "AVAILABLE"


def _publication(record: dict[str, object]) -> dict[str, object]:
    abstract = record.get("abstract")
    return {
        "title": str(record.get("title", ""))[:500],
        "doi": record.get("doi"),
        "year": record.get("year") or record.get("publication_date"),
        "authors": list(record.get("authors", []))[:20] if isinstance(record.get("authors"), list) else [],
        "venue": record.get("journal") or record.get("venue"),
        "citation_count": record.get("citation_count") or record.get("cited_by_count"),
        "abstract": str(abstract)[:800] if isinstance(abstract, str) else None,
        "authorship_confidence": record.get("authorship_confidence"),
        "selection_reasons": list(record.get("selection_reasons", []))[:5] if isinstance(record.get("selection_reasons"), list) else [],
        "sources": list(record.get("sources", []))[:8] if isinstance(record.get("sources"), list) else ["openalex"],
        "source_records": list(record.get("source_records", []))[:8] if isinstance(record.get("source_records"), list) else [{
            "source": "openalex", "source_record_id": record.get("openalex_url"),
        }],
    }


def _identity_evidence(result: dict[str, object]) -> list[dict[str, object]]:
    details = result.get("source_details", {})
    if not isinstance(details, dict):
        return []
    evidence = []
    for source, source_detail in details.items():
        if not isinstance(source_detail, dict):
            continue
        identities = source_detail.get("identities", [])
        for identity in identities[:3] if isinstance(identities, list) else []:
            if isinstance(identity, dict):
                evidence.append({
                    "source": source,
                    "name": identity.get("name"),
                    "aliases": list(identity.get("name_variants", []))[:10] if isinstance(identity.get("name_variants"), list) else [],
                    "affiliations": list(identity.get("affiliations", []))[:8] if isinstance(identity.get("affiliations"), list) else [],
                    "identifiers": identity.get("identifiers", {}),
                    "orcid": identity.get("orcid"),
                    "retrieved_at": source_detail.get("retrieved_at"),
                })
    return evidence[:15]


def _coverage(result: dict[str, object]) -> dict[str, dict[str, object]]:
    coverage = result.get("coverage", {})
    return {
        str(source): {
            "status": values.get("status"),
            "retrieved_publications": values.get("publication_count"),
            "reported_publications": values.get("reported_document_count"),
            "citation_count": values.get("citation_count"),
            "h_index": values.get("h_index"),
        }
        for source, values in coverage.items()
        if isinstance(values, dict)
    } if isinstance(coverage, dict) else {}


def _intelligence(query: str) -> dict[str, object]:
    result = academic_intelligence(_query(query))
    if not isinstance(result, dict):
        raise ValueError("Academic Intelligence returned an invalid result")
    return result


def _bounded(payload: dict[str, object]) -> dict[str, object]:
    if len(json.dumps(payload, ensure_ascii=False)) <= MAX_ACADEMIC_OUTPUT_CHARS:
        return payload
    for key in ("publications", "representative_papers"):
        records = payload.get(key)
        if isinstance(records, list):
            for record in records:
                if isinstance(record, dict):
                    record["abstract"] = None
    for key in ("identity_evidence", "coverage_conflicts", "publications", "representative_papers", "provenance"):
        records = payload.get(key)
        while isinstance(records, list) and len(records) > 1 and len(json.dumps(payload, ensure_ascii=False)) > MAX_ACADEMIC_OUTPUT_CHARS:
            records.pop()
    payload["truncated"] = True
    return payload


@ACADEMIC_MCP.tool(
    description="Resolve who a researcher is using independent identity evidence such as ORCID, curated author IDs, affiliations, aliases, and public scholarly graphs. This does not assume one provider defines the publication corpus.",
    structured_output=True,
)
def academic_resolve_researcher(query: str) -> dict[str, object]:
    result = _intelligence(query)
    return _bounded({
        "status": _status(result, require_identity=True),
        "researcher": result.get("researcher", {}),
        "identity_evidence": _identity_evidence(result),
        "identity_conflicts": [
            conflict for conflict in result.get("conflicts", [])
            if isinstance(conflict, dict) and conflict.get("type") in {"identity_unresolved", "affiliation_mismatch", "possible_split_profile"}
        ][:10],
        "provider_states": result.get("source_status", {}),
        "cache_hit": result.get("cache_hit", False),
    })


@ACADEMIC_MCP.tool(
    description="Search a bounded set of scholarly publication metadata for a topic. Use for scholarly evidence, not merely because a request mentions AI or technology. DOI and publisher pages can be verified later with Web or Browser.",
    structured_output=True,
)
def academic_search_publications(query: str, limit: int = 8) -> dict[str, object]:
    if not 1 <= limit <= 10:
        raise ValueError("limit must be between 1 and 10")
    diagnostics: list[dict[str, object]] = []
    publications = academic_papers((_query(query),), limit_per_query=limit, diagnostics=diagnostics)
    return _bounded({
        "status": "AVAILABLE" if publications else "DEGRADED",
        "publications": [_publication(item) for item in publications[:limit] if isinstance(item, dict)],
        "source_coverage": {"openalex": {"status": "AVAILABLE_FULL" if publications else "UNAVAILABLE", "result_count": len(publications)}},
        "provenance": [{"source": "openalex", **item} for item in diagnostics[:5]],
    })


@ACADEMIC_MCP.tool(
    description="Get a bounded researcher evidence package: resolved identity, per-source citation metrics, corpus summary, coverage conflicts, and representative papers selected from the verified multi-source corpus.",
    structured_output=True,
)
def academic_get_researcher_evidence(query: str) -> dict[str, object]:
    result = _intelligence(query)
    coverage = _coverage(result)
    return _bounded({
        "status": _status(result, require_identity=True),
        "researcher": result.get("researcher", {}),
        "citation_metrics_by_source": {
            source: {
                "publication_count": values.get("reported_publications") or values.get("retrieved_publications"),
                "citation_count": values.get("citation_count"),
                "h_index": values.get("h_index"),
            }
            for source, values in coverage.items()
        },
        "source_coverage": coverage,
        "coverage_conflicts": list(result.get("conflicts", []))[:15],
        "corpus": {
            "candidate_count": result.get("publication_candidate_count", 0),
            "verified_count": result.get("merged_publication_count", 0),
            "stored_in_cache": True,
        },
        "representative_papers": [
            _publication(item) for item in result.get("representative_papers", [])[:8]
            if isinstance(item, dict)
        ],
        "provider_states": result.get("source_status", {}),
        "cache_hit": result.get("cache_hit", False),
        "retrieved_at": result.get("retrieved_at"),
    })


@ACADEMIC_MCP.tool(
    description="Compare publication and citation coverage by source without collapsing conflicting counts into one absolute metric. Reports possible incomplete indexes, namesakes, and split profiles.",
    structured_output=True,
)
def academic_compare_source_coverage(query: str) -> dict[str, object]:
    result = _intelligence(query)
    return _bounded({
        "status": _status(result),
        "researcher": result.get("researcher", {}),
        "source_coverage": _coverage(result),
        "coverage_conflicts": list(result.get("conflicts", []))[:20],
        "provider_states": result.get("source_status", {}),
        "cache_hit": result.get("cache_hit", False),
        "retrieved_at": result.get("retrieved_at"),
    })


if __name__ == "__main__":
    ACADEMIC_MCP.run(transport="stdio")
