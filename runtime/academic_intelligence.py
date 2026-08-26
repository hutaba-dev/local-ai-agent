"""Multi-source researcher identity and publication intelligence."""

from __future__ import annotations

import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from enum import Enum
from time import monotonic
from typing import Callable, Iterable

import httpx


SCOPUS_API_BASE = "https://api.elsevier.com/content"
WOS_STARTER_API_BASE = "https://api.clarivate.com/apis/wos-starter/v1"
WOS_RESEARCHER_API_BASE = "https://api.clarivate.com/apis/wos-researcher"
OPENALEX_API_BASE = "https://api.openalex.org"
CROSSREF_API_BASE = "https://api.crossref.org"
S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
ORCID_API_BASE = "https://pub.orcid.org/v3.0"
USER_AGENT = "local-ai-agent-academic-intelligence/1.0"
METRICS_TTL_SECONDS = 6 * 60 * 60
TRANSIENT_FAILURE_TTL_SECONDS = 60


class SourceStatus(str, Enum):
    AVAILABLE_FULL = "AVAILABLE_FULL"
    AVAILABLE_LIMITED = "AVAILABLE_LIMITED"
    NO_ENTITLEMENT = "NO_ENTITLEMENT"
    RATE_LIMITED = "RATE_LIMITED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class AcademicSourceResult:
    source: str
    status: SourceStatus
    identities: tuple[dict[str, object], ...] = ()
    publications: tuple[dict[str, object], ...] = ()
    metrics: dict[str, object] = field(default_factory=dict)
    error: str | None = None
    details: dict[str, object] = field(default_factory=dict)

    def public_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["status"] = self.status.value
        return result


class _TTLCache:
    def __init__(self) -> None:
        self._entries: dict[str, tuple[float, object]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> object | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at <= monotonic():
                self._entries.pop(key, None)
                return None
            return value

    def set(self, key: str, value: object, ttl_seconds: int) -> None:
        with self._lock:
            self._entries[key] = (monotonic() + ttl_seconds, value)


_CACHE = _TTLCache()


def _request_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, object] | None = None,
    timeout: float = 15,
) -> dict[str, object]:
    response = httpx.get(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json", **(headers or {})},
        params=params,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("academic provider returned a non-object response")
    return payload


def _provider_failure(source: str, error: BaseException) -> AcademicSourceResult:
    if isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code
        if status_code == 429:
            status = SourceStatus.RATE_LIMITED
        elif status_code in {401, 403}:
            status = SourceStatus.NO_ENTITLEMENT
        else:
            status = SourceStatus.UNAVAILABLE
        reason = f"http_{status_code}"
    elif isinstance(error, httpx.TimeoutException):
        status = SourceStatus.UNAVAILABLE
        reason = "timeout"
    else:
        status = SourceStatus.UNAVAILABLE
        reason = "network_or_parse_error"
    return AcademicSourceResult(source, status, error=reason)


def _scopus_headers() -> dict[str, str] | None:
    api_key = _ascii_credential(os.getenv("SCOPUS_API_KEY"))
    if api_key is None:
        return None
    headers = {"X-ELS-APIKey": api_key}
    if institutional_token := _ascii_credential(os.getenv("SCOPUS_INST_TOKEN")):
        headers["X-ELS-Insttoken"] = institutional_token
    return headers


def _ascii_credential(value: str | None) -> str | None:
    candidate = value.strip() if isinstance(value, str) else ""
    if not candidate or candidate.casefold().startswith("replace-with-"):
        return None
    try:
        candidate.encode("ascii")
    except UnicodeEncodeError:
        return None
    return candidate


def scopus_search_authors(query: str, count: int = 10) -> dict[str, object]:
    headers = _scopus_headers()
    if headers is None:
        raise RuntimeError("SCOPUS_API_KEY is not configured")
    return _request_json(
        f"{SCOPUS_API_BASE}/search/author",
        headers=headers,
        params={"query": query, "count": min(max(count, 1), 25), "view": "STANDARD"},
    )


def scopus_get_author(author_id: str) -> dict[str, object]:
    headers = _scopus_headers()
    if headers is None:
        raise RuntimeError("SCOPUS_API_KEY is not configured")
    return _request_json(
        f"{SCOPUS_API_BASE}/author/author_id/{author_id}",
        headers=headers,
        params={"view": "ENHANCED"},
    )


def scopus_get_author_documents(author_id: str, count: int = 100) -> dict[str, object]:
    return scopus_search_documents(f"AU-ID({author_id})", count=count)


def scopus_search_documents(query: str, count: int = 25) -> dict[str, object]:
    headers = _scopus_headers()
    if headers is None:
        raise RuntimeError("SCOPUS_API_KEY is not configured")
    return _request_json(
        f"{SCOPUS_API_BASE}/search/scopus",
        headers=headers,
        params={"query": query, "count": min(max(count, 1), 200), "view": "STANDARD"},
    )


def scopus_get_abstract(*, scopus_id: str | None = None, doi: str | None = None) -> dict[str, object]:
    headers = _scopus_headers()
    if headers is None:
        raise RuntimeError("SCOPUS_API_KEY is not configured")
    if scopus_id:
        path = f"abstract/scopus_id/{scopus_id}"
    elif doi:
        path = f"abstract/doi/{doi}"
    else:
        raise ValueError("scopus_id or doi is required")
    return _request_json(f"{SCOPUS_API_BASE}/{path}", headers=headers, params={"view": "FULL"})


def scopus_get_citation_overview(scopus_ids: Iterable[str], date_range: str | None = None) -> dict[str, object]:
    headers = _scopus_headers()
    if headers is None:
        raise RuntimeError("SCOPUS_API_KEY is not configured")
    identifiers = [identifier for identifier in scopus_ids if identifier][:25]
    if not identifiers:
        raise ValueError("at least one Scopus ID is required")
    params: dict[str, object] = {"scopus_id": ",".join(identifiers)}
    if date_range:
        params["date"] = date_range
    return _request_json(f"{SCOPUS_API_BASE}/abstract/citations", headers=headers, params=params)


def _wos_headers() -> dict[str, str] | None:
    api_key = os.getenv("WOS_API_KEY")
    return {"X-ApiKey": api_key} if api_key else None


def wos_search_researchers(query: str, limit: int = 10, page: int = 1) -> dict[str, object]:
    headers = _wos_headers()
    if headers is None:
        raise RuntimeError("WOS_API_KEY is not configured")
    return _request_json(
        f"{WOS_RESEARCHER_API_BASE}/researchers",
        headers=headers,
        params={"q": query, "limit": min(max(limit, 1), 50), "page": max(page, 1)},
    )


def wos_get_researcher(researcher_id: str) -> dict[str, object]:
    headers = _wos_headers()
    if headers is None:
        raise RuntimeError("WOS_API_KEY is not configured")
    return _request_json(f"{WOS_RESEARCHER_API_BASE}/researchers/{researcher_id}", headers=headers)


def wos_search_documents(query: str, limit: int = 50, page: int = 1) -> dict[str, object]:
    headers = _wos_headers()
    if headers is None:
        raise RuntimeError("WOS_API_KEY is not configured")
    return _request_json(
        f"{WOS_STARTER_API_BASE}/documents",
        headers=headers,
        params={"q": query, "limit": min(max(limit, 1), 50), "page": max(page, 1)},
    )


def wos_get_document(uid: str) -> dict[str, object]:
    headers = _wos_headers()
    if headers is None:
        raise RuntimeError("WOS_API_KEY is not configured")
    return _request_json(f"{WOS_STARTER_API_BASE}/documents/{uid}", headers=headers)


def wos_get_citation_metrics(query: str, limit: int = 50) -> dict[str, object]:
    """WoS Starter returns per-document times-cited when the plan permits it."""
    return wos_search_documents(query, limit=limit)


def _name_from_query(query: str) -> str:
    quoted_korean = re.search(r'["“”]([가-힣]{2,5})["“”]', query)
    if quoted_korean:
        return quoted_korean.group(1)
    korean = re.search(r"([가-힣]{2,5})\s*(?:교수|박사|연구자|학자)", query)
    if korean:
        return korean.group(1)
    cleaned = re.sub(
        r"\s+(?:professor|researcher|academic|scientist|ph\.?d\.?)\s*$",
        " ",
        query,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    romanized = re.search(
        r"\b([A-Z][A-Za-z'-]+(?:\s+[A-Z][A-Za-z'-]+){1,3})",
        cleaned,
        flags=re.IGNORECASE,
    )
    if romanized:
        return romanized.group(1)
    return cleaned[:160]


def _academic_aliases(query: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(
        alias.strip()
        for alias in re.findall(r"^Academic alias:\s*([^\n]+)$", query, re.MULTILINE | re.IGNORECASE)
        if alias.strip()
    ))


def _affiliation_hint(query: str) -> str | None:
    match = re.search(r"^Affiliation hint:\s*([^\n]+)$", query, re.MULTILINE | re.IGNORECASE)
    return match.group(1).strip() if match and match.group(1).strip() else None


def _lookup_name(query: str) -> str:
    aliases = _academic_aliases(query)
    return aliases[0] if aliases else _name_from_query(query)


def _scopus_author_entries(payload: dict[str, object]) -> list[dict[str, object]]:
    search_results = payload.get("search-results")
    entries = search_results.get("entry", []) if isinstance(search_results, dict) else []
    normalized = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        preferred = entry.get("preferred-name")
        indexed_name = None
        if isinstance(preferred, dict):
            indexed_name = preferred.get("ce:indexed-name") or preferred.get("indexed-name")
            if not indexed_name:
                given_name = str(preferred.get("given-name", "")).strip()
                surname = str(preferred.get("surname", "")).strip()
                indexed_name = " ".join(value for value in (given_name, surname) if value)
        orcid = entry.get("orcid")
        if isinstance(orcid, str):
            orcid = orcid.strip("[]")
        identity = {
            "name": indexed_name or entry.get("dc:title"),
            "source": "scopus",
            "identifiers": {"scopus_author_id": entry.get("dc:identifier", "").removeprefix("AUTHOR_ID:") if isinstance(entry.get("dc:identifier"), str) else None},
            "affiliations": [entry.get("affiliation-current", {}).get("affiliation-name")] if isinstance(entry.get("affiliation-current"), dict) else [],
            "document_count": _to_int(entry.get("document-count")),
            "orcid": orcid,
        }
        if identity["name"] and identity["identifiers"]["scopus_author_id"]:
            normalized.append(identity)
    return normalized


def _scopus_documents(payload: dict[str, object]) -> list[dict[str, object]]:
    search_results = payload.get("search-results")
    entries = search_results.get("entry", []) if isinstance(search_results, dict) else []
    return [
        {
            "title": entry.get("dc:title"), "doi": entry.get("prism:doi"),
            "year": _year(entry.get("prism:coverDate")), "journal": entry.get("prism:publicationName"),
            "authors": [entry.get("dc:creator")] if entry.get("dc:creator") else [],
            "citation_count": _to_int(entry.get("citedby-count")),
            "source_ids": {"scopus_id": entry.get("dc:identifier")}, "sources": ["scopus"],
        }
        for entry in entries if isinstance(entry, dict) and entry.get("dc:title")
    ]


def _scopus_provider(query: str) -> AcademicSourceResult:
    if _scopus_headers() is None:
        return AcademicSourceResult("scopus", SourceStatus.UNAVAILABLE, error="SCOPUS_API_KEY is not configured")
    names = _academic_aliases(query) or (_name_from_query(query),)
    try:
        identities: list[dict[str, object]] = []
        searched_names: list[str] = []
        affiliation_hint = _affiliation_hint(query)
        for name in names:
            searched_names.append(name)
            identities.extend(_scopus_author_entries(scopus_search_authors(_scopus_author_query(name))))
            if any(
                _scopus_affiliation_matches(identity, affiliation_hint)
                and (_to_int(identity.get("document_count")) or 0) >= 5
                for identity in identities
            ):
                break
        identities = list({
            str(identity.get("identifiers", {}).get("scopus_author_id")): identity
            for identity in identities
            if isinstance(identity.get("identifiers"), dict)
        }.values())
        identities = _select_scopus_identities(identities, affiliation_hint)
        publications: list[dict[str, object]] = []
        metrics: dict[str, object] = {}
        details: dict[str, object] = {"searched_names": searched_names}
        status = SourceStatus.AVAILABLE_LIMITED
        if len(identities) == 1:
            author_id = identities[0].get("identifiers", {}).get("scopus_author_id")
            if isinstance(author_id, str) and author_id:
                try:
                    profile_identity, metrics = _scopus_author_profile(scopus_get_author(author_id), identities[0])
                    identities = [profile_identity]
                    publications = _scopus_documents(scopus_get_author_documents(author_id))
                    status = SourceStatus.AVAILABLE_FULL
                except httpx.HTTPStatusError as error:
                    if error.response.status_code not in {401, 403}:
                        raise
                    details["profile_or_documents"] = f"http_{error.response.status_code}"
        return AcademicSourceResult("scopus", status, tuple(identities), tuple(publications), metrics, details=details)
    except (httpx.HTTPError, RuntimeError, ValueError) as error:
        return _provider_failure("scopus", error)


def _select_scopus_identities(
    identities: list[dict[str, object]], affiliation_hint: str | None
) -> list[dict[str, object]]:
    candidates = identities
    if affiliation_hint:
        affiliation_matches = [
            identity for identity in candidates
            if _scopus_affiliation_matches(identity, affiliation_hint)
        ]
        if affiliation_matches:
            candidates = affiliation_matches
    candidates.sort(key=lambda identity: _to_int(identity.get("document_count")) or 0, reverse=True)
    if len(candidates) <= 1:
        return candidates
    leading = _to_int(candidates[0].get("document_count")) or 0
    runner_up = _to_int(candidates[1].get("document_count")) or 0
    return candidates[:1] if leading >= 5 and leading >= max(1, runner_up) * 2 else candidates


def _scopus_affiliation_matches(identity: dict[str, object], affiliation_hint: str | None) -> bool:
    if not affiliation_hint:
        return False
    hint_terms = set(_normalize_name(affiliation_hint).split())
    return any(
        hint_terms <= set(_normalize_name(affiliation).split())
        or set(_normalize_name(affiliation).split()) <= hint_terms
        for affiliation in identity.get("affiliations", [])
        if isinstance(affiliation, str)
    )


def _wos_provider(query: str) -> AcademicSourceResult:
    if _wos_headers() is None:
        return AcademicSourceResult("web_of_science", SourceStatus.UNAVAILABLE, error="WOS_API_KEY is not configured")
    name = _lookup_name(query)
    researcher_error: str | None = None
    identities: list[dict[str, object]] = []
    try:
        researcher_payload = wos_search_researchers(name)
        identities = _wos_researcher_entries(researcher_payload)
    except httpx.HTTPStatusError as error:
        if error.response.status_code in {401, 403, 404}:
            researcher_error = f"researcher_api_http_{error.response.status_code}"
        else:
            return _provider_failure("web_of_science", error)
    except (httpx.HTTPError, RuntimeError, ValueError) as error:
        researcher_error = type(error).__name__
    try:
        documents_payload = wos_search_documents(f'AU=("{name}")')
        publications = _wos_documents(documents_payload)
        status = SourceStatus.AVAILABLE_FULL if any(work.get("citation_count") is not None for work in publications) else SourceStatus.AVAILABLE_LIMITED
        metrics = {"document_count": len(publications)}
        return AcademicSourceResult(
            "web_of_science", status, tuple(identities), tuple(publications), metrics,
            details={"researcher_api": researcher_error or "available"},
        )
    except (httpx.HTTPError, RuntimeError, ValueError) as error:
        return _provider_failure("web_of_science", error)


def _openalex_provider(query: str) -> AcademicSourceResult:
    name = _lookup_name(query)
    try:
        author_payload = _request_json(
            f"{OPENALEX_API_BASE}/authors",
            params={"search": name, "per-page": 5, "mailto": os.getenv("UNPAYWALL_EMAIL", "")},
        )
        candidates = author_payload.get("results", [])
        identities = [_openalex_identity(item) for item in candidates if isinstance(item, dict)]
        publications: list[dict[str, object]] = []
        metrics: dict[str, object] = {}
        exact = [candidate for candidate in identities if _normalize_name(str(candidate.get("name", ""))) == _normalize_name(name)]
        if len(exact) == 1:
            author_id = exact[0].get("identifiers", {}).get("openalex_author_id")
            if isinstance(author_id, str):
                works_payload = _request_json(
                    f"{OPENALEX_API_BASE}/works",
                    params={"filter": f"author.id:{author_id}", "per-page": 100, "sort": "cited_by_count:desc"},
                )
                publications = [_openalex_work(work) for work in works_payload.get("results", []) if isinstance(work, dict)]
                metrics = {
                    "document_count": exact[0].get("document_count"),
                    "citation_count": exact[0].get("citation_count"),
                    "h_index": exact[0].get("h_index"),
                }
        return AcademicSourceResult("openalex", SourceStatus.AVAILABLE_FULL, tuple(identities), tuple(publications), metrics)
    except (httpx.HTTPError, RuntimeError, ValueError) as error:
        return _provider_failure("openalex", error)


def _semantic_scholar_provider(query: str) -> AcademicSourceResult:
    name = _lookup_name(query)
    headers = {"x-api-key": os.getenv("S2_API_KEY", "")} if os.getenv("S2_API_KEY") else {}
    try:
        payload = _request_json(
            f"{S2_API_BASE}/author/search", headers=headers,
            params={"query": name, "limit": 5, "fields": "name,paperCount,citationCount,hIndex,affiliations"},
        )
        identities = []
        for item in payload.get("data", []):
            if not isinstance(item, dict):
                continue
            identities.append({
                "name": item.get("name"), "source": "semantic_scholar",
                "identifiers": {"semantic_scholar_author_id": item.get("authorId")},
                "affiliations": item.get("affiliations", []), "document_count": item.get("paperCount"),
                "citation_count": item.get("citationCount"), "h_index": item.get("hIndex"),
            })
        exact = [
            identity for identity in identities
            if _normalize_name(str(identity.get("name", ""))) == _normalize_name(name)
        ]
        publications: list[dict[str, object]] = []
        metrics: dict[str, object] = {}
        status = SourceStatus.AVAILABLE_LIMITED
        if len(exact) == 1:
            author_id = exact[0].get("identifiers", {}).get("semantic_scholar_author_id")
            if isinstance(author_id, str) and author_id:
                papers_payload = _request_json(
                    f"{S2_API_BASE}/author/{author_id}/papers",
                    headers=headers,
                    params={
                        "limit": 100,
                        "fields": "title,year,citationCount,externalIds,venue,authors,abstract",
                    },
                )
                publications = [
                    _semantic_scholar_work(item)
                    for item in papers_payload.get("data", [])
                    if isinstance(item, dict) and item.get("title")
                ]
                metrics = {
                    "document_count": exact[0].get("document_count"),
                    "citation_count": exact[0].get("citation_count"),
                    "h_index": exact[0].get("h_index"),
                }
                status = SourceStatus.AVAILABLE_FULL
        return AcademicSourceResult(
            "semantic_scholar", status, tuple(identities), tuple(publications), metrics
        )
    except (httpx.HTTPError, RuntimeError, ValueError) as error:
        return _provider_failure("semantic_scholar", error)


def _semantic_scholar_work(item: dict[str, object]) -> dict[str, object]:
    external_ids = item.get("externalIds")
    identifiers = external_ids if isinstance(external_ids, dict) else {}
    authors = item.get("authors")
    return {
        "title": item.get("title"),
        "doi": identifiers.get("DOI"),
        "year": _to_int(item.get("year")),
        "journal": item.get("venue"),
        "authors": [
            str(author.get("name")) for author in authors
            if isinstance(author, dict) and author.get("name")
        ] if isinstance(authors, list) else [],
        "abstract": item.get("abstract"),
        "citation_count": _to_int(item.get("citationCount")),
        "source_ids": {
            "semantic_scholar_paper_id": item.get("paperId"),
            **{str(key).lower(): value for key, value in identifiers.items() if value},
        },
        "sources": ["semantic_scholar"],
    }


def _orcid_provider(query: str) -> AcademicSourceResult:
    name = _lookup_name(query)
    try:
        payload = _request_json(
            f"{ORCID_API_BASE}/expanded-search/",
            params={"q": f'given-and-family-names:\"{name}\"', "rows": 10},
        )
        identities = []
        for item in payload.get("expanded-result", []):
            if not isinstance(item, dict):
                continue
            display_name = " ".join(
                str(item.get(field, "")).strip()
                for field in ("given-names", "family-names")
            ).strip()
            orcid = item.get("orcid-id")
            if not display_name or not isinstance(orcid, str):
                continue
            affiliations = item.get("institution-name", [])
            identities.append({
                "name": display_name,
                "source": "orcid",
                "identifiers": {"orcid": orcid},
                "affiliations": affiliations if isinstance(affiliations, list) else [],
            })
        status = SourceStatus.AVAILABLE_FULL if identities else SourceStatus.AVAILABLE_LIMITED
        return AcademicSourceResult("orcid", status, tuple(identities))
    except (httpx.HTTPError, RuntimeError, ValueError) as error:
        return _provider_failure("orcid", error)


def _crossref_provider(query: str) -> AcademicSourceResult:
    name = _lookup_name(query)
    try:
        payload = _request_json(
            f"{CROSSREF_API_BASE}/works",
            params={"query.author": name, "rows": 20, "select": "DOI,title,author,published,container-title,is-referenced-by-count"},
        )
        message = payload.get("message")
        items = message.get("items", []) if isinstance(message, dict) else []
        publications = [
            publication for work in items if isinstance(work, dict)
            for publication in [_crossref_work(work)]
            if any(
                _name_similarity(str(author), name) >= 0.8
                for author in publication.get("authors", [])
                if isinstance(author, str)
            )
        ]
        return AcademicSourceResult("crossref", SourceStatus.AVAILABLE_FULL, publications=tuple(publications))
    except (httpx.HTTPError, RuntimeError, ValueError) as error:
        return _provider_failure("crossref", error)


def _google_scholar_provider(query: str) -> AcademicSourceResult:
    name = _lookup_name(query)
    try:
        from runtime.web_search import search_many

        results = search_many((f'site:scholar.google.com/citations "{name}"', f'"{name}" Google Scholar'), "DEEP_RESEARCH")
        profiles = [
            {"title": item.get("title"), "url": item.get("url"), "description": item.get("description")}
            for item in results if isinstance(item, dict) and "scholar.google" in str(item.get("url", ""))
        ]
        status = SourceStatus.AVAILABLE_LIMITED if profiles else SourceStatus.UNAVAILABLE
        return AcademicSourceResult(
            "google_scholar", status,
            identities=tuple({"name": name, "source": "google_scholar", "identifiers": {"google_scholar_profile": profile["url"]}, "profile": profile} for profile in profiles[:3]),
            error=None if profiles else "no public Scholar profile discovered",
            details={"method": "public_profile_discovery_via_web_search", "scraping": False},
        )
    except (httpx.HTTPError, RuntimeError, ValueError) as error:
        return _provider_failure("google_scholar", error)


def academic_source_status() -> dict[str, str]:
    return {
        "scopus": SourceStatus.AVAILABLE_LIMITED.value if _scopus_headers() else SourceStatus.UNAVAILABLE.value,
        "web_of_science": SourceStatus.AVAILABLE_LIMITED.value if _wos_headers() else SourceStatus.UNAVAILABLE.value,
        "google_scholar": SourceStatus.AVAILABLE_LIMITED.value if os.getenv("BRAVE_SEARCH_API_KEY") else SourceStatus.UNAVAILABLE.value,
        "openalex": SourceStatus.AVAILABLE_FULL.value,
        "semantic_scholar": SourceStatus.AVAILABLE_LIMITED.value,
        "orcid": SourceStatus.AVAILABLE_FULL.value,
        "crossref": SourceStatus.AVAILABLE_FULL.value,
    }


def academic_intelligence(query: str) -> dict[str, object]:
    identity_names = (_name_from_query(query), *_academic_aliases(query))
    cache_key = f"academic-intelligence:{'|'.join(_normalize_name(name) for name in identity_names)}"
    cached = _CACHE.get(cache_key)
    if isinstance(cached, dict):
        return cached | {"cache_hit": True}
    curated: dict[str, Callable[[str], AcademicSourceResult]] = {}
    results: list[AcademicSourceResult] = []
    if _scopus_headers():
        curated["scopus"] = _scopus_provider
    else:
        results.append(AcademicSourceResult("scopus", SourceStatus.UNAVAILABLE, error="SCOPUS_API_KEY is not configured"))
    if _wos_headers():
        curated["web_of_science"] = _wos_provider
    else:
        results.append(AcademicSourceResult("web_of_science", SourceStatus.UNAVAILABLE, error="WOS_API_KEY is not configured"))
    if os.getenv("BRAVE_SEARCH_API_KEY"):
        curated["google_scholar"] = _google_scholar_provider
    else:
        results.append(AcademicSourceResult("google_scholar", SourceStatus.UNAVAILABLE, error="BRAVE_SEARCH_API_KEY is not configured"))
    if curated:
        results.extend(_run_providers(query, curated))
    fallback: dict[str, Callable[[str], AcademicSourceResult]] = {
        "openalex": _openalex_provider,
        "semantic_scholar": _semantic_scholar_provider,
        "orcid": _orcid_provider,
        "crossref": _crossref_provider,
    }
    paid_results = [result for result in results if result.source in {"scopus", "web_of_science"}]
    public_fallback_triggered = len(paid_results) < 2 or not _curated_sources_agree(paid_results)
    if public_fallback_triggered:
        results.extend(_run_providers(query, fallback))
    intelligence = _aggregate_intelligence(query, results)
    intelligence["selection_policy"] = {
        "curated_first": True,
        "public_fallback_triggered": public_fallback_triggered,
        "providers_called": [*curated, *(fallback if public_fallback_triggered else {})],
    }
    cache_ttl = (
        TRANSIENT_FAILURE_TTL_SECONDS
        if SourceStatus.RATE_LIMITED.value in intelligence.get("source_status", {}).values()
        else METRICS_TTL_SECONDS
    )
    _CACHE.set(cache_key, intelligence, cache_ttl)
    return intelligence | {"cache_hit": False}


def _run_providers(
    query: str, providers: dict[str, Callable[[str], AcademicSourceResult]]
) -> list[AcademicSourceResult]:
    results: list[AcademicSourceResult] = []
    with ThreadPoolExecutor(max_workers=max(1, len(providers)), thread_name_prefix="academic-source") as executor:
        futures = {executor.submit(provider, query): source for source, provider in providers.items()}
        for future in as_completed(futures):
            source = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                results.append(AcademicSourceResult(source, SourceStatus.UNAVAILABLE, error=type(error).__name__))
    return results


def _curated_sources_agree(results: list[AcademicSourceResult]) -> bool:
    if len(results) != 2 or any(
        result.status not in {SourceStatus.AVAILABLE_FULL, SourceStatus.AVAILABLE_LIMITED}
        or not result.identities
        for result in results
    ):
        return False
    counts = [_to_int(result.metrics.get("document_count")) for result in results]
    if any(count is None or count <= 0 for count in counts):
        return False
    smaller, larger = sorted(counts)  # type: ignore[arg-type]
    return smaller / larger >= 0.75


def _aggregate_intelligence(query: str, results: list[AcademicSourceResult]) -> dict[str, object]:
    target_name = _name_from_query(query)
    aliases = _academic_aliases(query)
    identity = _resolve_identity(target_name, aliases, results)
    candidate_corpus = _merge_publications(results, identity["confidence"], (target_name, *aliases))
    verified_corpus = [
        publication for publication in candidate_corpus
        if publication.get("authorship_confidence") in {"HIGH", "MEDIUM"}
    ]
    coverage = {
        result.source: {
            "status": result.status.value,
            "publication_count": len(result.publications),
            "reported_document_count": result.metrics.get("document_count"),
            "citation_count": result.metrics.get("citation_count"),
            "h_index": result.metrics.get("h_index"),
        }
        for result in results
    }
    conflicts = _coverage_conflicts(coverage, identity, results)
    return {
        "researcher": identity,
        "source_status": {result.source: result.status.value for result in results},
        "source_details": {
            result.source: {
                "source": result.source,
                "status": result.status.value,
                "identities": list(result.identities[:5]),
                "publications": list(result.publications[:20]),
                "metrics": result.metrics,
                "error": result.error,
                "details": result.details,
            }
            for result in results
        },
        "coverage": coverage,
        "conflicts": conflicts,
        "publication_candidates": candidate_corpus,
        "publication_candidate_count": len(candidate_corpus),
        "merged_verified_corpus": verified_corpus,
        "merged_publication_count": len(verified_corpus),
        "representative_papers": _representative_papers(verified_corpus),
        "pipeline": [
            "IDENTITY_RESOLUTION", "AUTHOR_IDENTIFIER_RESOLUTION", "MULTI_SOURCE_PUBLICATION_DISCOVERY",
            "DEDUPLICATION", "AUTHORSHIP_VERIFICATION", "COVERAGE_CHECK", "CITATION_METRIC_CROSS_CHECK",
            "REPRESENTATIVE_PAPER_SELECTION",
        ],
    }


def _resolve_identity(
    target_name: str, aliases: tuple[str, ...], results: list[AcademicSourceResult]
) -> dict[str, object]:
    matching: list[dict[str, object]] = []
    for result in results:
        for identity in result.identities:
            name = identity.get("name")
            name_variants = identity.get("name_variants")
            candidate_identity_names = [name] if isinstance(name, str) else []
            if isinstance(name_variants, list):
                candidate_identity_names.extend(
                    variant for variant in name_variants if isinstance(variant, str)
                )
            if any(
                _name_similarity(identity_name, candidate_name) >= 0.8
                for identity_name in candidate_identity_names
                for candidate_name in (target_name, *aliases)
            ):
                matching.append(identity)
    sources = sorted({str(identity.get("source")) for identity in matching})
    identifiers: dict[str, object] = {
        "orcid": None, "scopus_author_id": None, "wos_researcher_id": None,
        "openalex_author_id": None, "semantic_scholar_author_id": None,
        "google_scholar_profile": None,
    }
    for identity in matching:
        values = identity.get("identifiers")
        if isinstance(values, dict):
            for key, value in values.items():
                if key in identifiers and value and identifiers[key] is None:
                    identifiers[key] = value
        if identity.get("orcid") and identifiers["orcid"] is None:
            identifiers["orcid"] = identity["orcid"]
    affiliations = sorted({
        affiliation for identity in matching
        for affiliation in identity.get("affiliations", [])
        if isinstance(affiliation, str) and affiliation
    })
    exact_names = {_normalize_name(str(identity.get("name", ""))) for identity in matching}
    if len(sources) >= 2 and len(exact_names) == 1 and affiliations and any(identifiers.values()):
        confidence = "HIGH"
    elif len(sources) >= 2 and len(exact_names) <= 2:
        confidence = "MEDIUM"
    elif len(matching) == 1:
        only_identity = matching[0]
        only_identifiers = only_identity.get("identifiers")
        confidence = "MEDIUM" if (
            only_identity.get("source") in {"scopus", "web_of_science"}
            and affiliations
            and (
                only_identity.get("orcid")
                or isinstance(only_identifiers, dict) and any(only_identifiers.values())
            )
        ) else "LOW"
    else:
        confidence = "AMBIGUOUS" if matching else "UNRESOLVED"
    return {
        "canonical_name": target_name,
        "aliases": list(aliases),
        "native_name": target_name if re.search(r"[가-힣]", target_name) else None,
        "affiliations": affiliations[:8],
        "identifiers": identifiers,
        "identity_confidence": confidence,
        "confidence": confidence,
        "identity_sources": sources,
        "candidate_count": len(matching),
    }


def _merge_publications(
    results: list[AcademicSourceResult], identity_confidence: object, target_names: tuple[str, ...]
) -> list[dict[str, object]]:
    merged: dict[str, dict[str, object]] = {}
    for result in results:
        for publication in result.publications:
            title = str(publication.get("title", "")).strip()
            if not title:
                continue
            key = _publication_key(publication)
            existing = merged.get(key)
            if existing is None:
                existing = dict(publication)
                existing["sources"] = sorted(set(publication.get("sources", [])) | {result.source})
                existing["source_ids"] = dict(publication.get("source_ids", {}))
                existing["verified_author_profile_source"] = (
                    result.source in {"scopus", "web_of_science"} and len(result.identities) == 1
                )
                merged[key] = existing
            else:
                existing["sources"] = sorted(set(existing.get("sources", [])) | set(publication.get("sources", [])) | {result.source})
                existing.setdefault("source_ids", {}).update(publication.get("source_ids", {}))
                existing["verified_author_profile_source"] = bool(
                    existing.get("verified_author_profile_source")
                    or result.source in {"scopus", "web_of_science"} and len(result.identities) == 1
                )
                for field_name in ("doi", "year", "journal", "authors", "abstract", "citation_count"):
                    if not existing.get(field_name) and publication.get(field_name):
                        existing[field_name] = publication[field_name]
    for publication in merged.values():
        source_count = len(publication.get("sources", []))
        authors = publication.get("authors")
        target_author_match = isinstance(authors, list) and any(
            isinstance(author, str) and any(
                _name_similarity(author, target_name) >= 0.8 for target_name in target_names
            )
            for author in authors
        )
        publication["authorship_confidence"] = (
            "HIGH" if source_count >= 2 and identity_confidence in {"HIGH", "MEDIUM"}
            else "MEDIUM" if source_count >= 2
            or (
                identity_confidence in {"HIGH", "MEDIUM"}
                and (
                    bool(publication.get("verified_author_profile_source"))
                    or target_author_match and bool(publication.get("doi"))
                )
            )
            else "LOW"
        )
    return sorted(
        merged.values(),
        key=lambda publication: (_to_int(publication.get("citation_count")) or 0, _to_int(publication.get("year")) or 0),
        reverse=True,
    )[:200]


def _publication_key(publication: dict[str, object]) -> str:
    doi = publication.get("doi")
    if isinstance(doi, str) and doi.strip():
        return f"doi:{doi.lower().removeprefix('https://doi.org/')}"
    title = _normalize_title(str(publication.get("title", "")))
    year = publication.get("year")
    authors = publication.get("authors")
    author_key = "|".join(sorted(_normalize_name(str(author)) for author in authors)) if isinstance(authors, list) else ""
    return f"title:{title}:{year or ''}:{author_key[:200]}"


def _coverage_conflicts(
    coverage: dict[str, dict[str, object]],
    identity: dict[str, object],
    results: list[AcademicSourceResult],
) -> list[dict[str, object]]:
    conflicts: list[dict[str, object]] = []
    counts = {
        source: _to_int(values.get("reported_document_count")) or _to_int(values.get("publication_count"))
        for source, values in coverage.items()
        if values.get("status") in {SourceStatus.AVAILABLE_FULL.value, SourceStatus.AVAILABLE_LIMITED.value}
    }
    positive = {source: count for source, count in counts.items() if isinstance(count, int) and count > 0}
    if len(positive) >= 2:
        largest_source, largest = max(positive.items(), key=lambda item: item[1])
        for source, count in positive.items():
            if count * 2 < largest and largest - count >= 10:
                conflicts.append({
                    "type": "publication_count_discrepancy", "source": source, "value": count,
                    "reference_source": largest_source, "reference_value": largest,
                    "assessment": "incomplete_split_or_misresolved_record",
                })
    for metric, minimum_gap, factor in (("citation_count", 100, 3), ("h_index", 5, 2)):
        values = {
            source: _to_int(details.get(metric)) for source, details in coverage.items()
            if _to_int(details.get(metric)) is not None
        }
        if len(values) >= 2:
            low_source, low = min(values.items(), key=lambda item: item[1])
            high_source, high = max(values.items(), key=lambda item: item[1])
            if high - low >= minimum_gap and high >= max(1, low) * factor:
                conflicts.append({
                    "type": f"{metric}_discrepancy", "source": low_source, "value": low,
                    "reference_source": high_source, "reference_value": high,
                    "assessment": "database_coverage_or_identity_conflict",
                })
    affiliation_sets = {
        result.source: {
            _normalize_name(affiliation)
            for candidate in result.identities
            for affiliation in candidate.get("affiliations", [])
            if isinstance(affiliation, str) and affiliation
        }
        for result in results
    }
    nonempty_affiliations = {source: values for source, values in affiliation_sets.items() if values}
    if len(nonempty_affiliations) >= 2 and not set.intersection(*nonempty_affiliations.values()):
        conflicts.append({"type": "affiliation_mismatch", "assessment": "identity_verification_required"})
    if identity.get("identity_confidence") in {"LOW", "AMBIGUOUS", "UNRESOLVED"}:
        conflicts.append({"type": "identity_unresolved", "assessment": "additional_identifier_verification_required"})
    return conflicts


def _representative_papers(corpus: list[dict[str, object]]) -> list[dict[str, object]]:
    if not corpus:
        return []
    by_citations = corpus[:5]
    recent = sorted(corpus, key=lambda item: _to_int(item.get("year")) or 0, reverse=True)[:3]
    selected: list[dict[str, object]] = []
    seen: set[str] = set()
    for paper in (*by_citations, *recent):
        key = _publication_key(paper)
        if key not in seen:
            seen.add(key)
            selected.append(paper)
    return selected[:8]


def _scopus_author_query(name: str) -> str:
    parts = name.replace(",", " ").split()
    if len(parts) >= 2:
        given_names = " ".join(parts[:-1])
        return f'AUTHLASTNAME({parts[-1]}) AND AUTHFIRST("{given_names}")'
    return f"AUTHLASTNAME({name})"


def _scopus_author_profile(
    payload: dict[str, object], fallback: dict[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    records = payload.get("author-retrieval-response")
    record = records[0] if isinstance(records, list) and records and isinstance(records[0], dict) else {}
    core = record.get("coredata") if isinstance(record.get("coredata"), dict) else {}
    profile = record.get("author-profile") if isinstance(record.get("author-profile"), dict) else {}
    preferred = profile.get("preferred-name") if isinstance(profile.get("preferred-name"), dict) else {}
    profile_name = preferred.get("indexed-name") or preferred.get("ce:indexed-name")
    fallback_name = fallback.get("name")
    selected_name = (
        profile_name
        if isinstance(profile_name, str)
        and _name_detail_score(profile_name) >= _name_detail_score(str(fallback_name or ""))
        else fallback_name
    )
    current = profile.get("affiliation-current")
    current_items = current.get("affiliation") if isinstance(current, dict) else []
    if isinstance(current_items, dict):
        current_items = [current_items]
    affiliations = [
        item.get("ip-doc", {}).get("afdispname") or item.get("affiliation-name")
        for item in current_items if isinstance(item, dict)
    ] if isinstance(current_items, list) else []
    identity = dict(fallback)
    identity.update({
        "name": selected_name,
        "name_variants": list(dict.fromkeys(
            value for value in (fallback_name, profile_name, *_scopus_name_variants(profile))
            if isinstance(value, str) and value
        )),
        "affiliations": [value for value in affiliations if isinstance(value, str) and value] or fallback.get("affiliations", []),
        "orcid": core.get("orcid") or profile.get("orcid") or fallback.get("orcid"),
        "subject_areas": record.get("subject-areas", {}).get("subject-area", []) if isinstance(record.get("subject-areas"), dict) else [],
    })
    metrics = {
        "document_count": _to_int(core.get("document-count") or fallback.get("document_count")),
        "citation_count": _to_int(core.get("citation-count")),
        "h_index": _to_int(record.get("h-index") or core.get("h-index")),
    }
    return identity, metrics


def _name_detail_score(value: str) -> int:
    return sum(len(term) for term in re.findall(r"[A-Za-z가-힣]+", value) if len(term) > 1)


def _scopus_name_variants(profile: dict[str, object]) -> list[str]:
    variants = profile.get("name-variant") or profile.get("name-variants") or []
    if isinstance(variants, dict):
        variants = variants.get("name-variant", [])
    if isinstance(variants, dict):
        variants = [variants]
    return [
        value for item in variants if isinstance(item, dict)
        for value in [item.get("indexed-name") or item.get("ce:indexed-name")]
        if isinstance(value, str)
    ] if isinstance(variants, list) else []


def _wos_researcher_entries(payload: dict[str, object]) -> list[dict[str, object]]:
    items = payload.get("hits") or payload.get("researchers") or payload.get("data") or []
    normalized = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        normalized.append({
            "name": item.get("displayName") or item.get("name"), "source": "web_of_science",
            "identifiers": {"wos_researcher_id": item.get("rid") or item.get("researcherId") or item.get("id")},
            "affiliations": item.get("affiliations", []), "document_count": item.get("documentsCount"),
            "citation_count": item.get("timesCited"), "h_index": item.get("hIndex"),
        })
    return normalized


def _wos_documents(payload: dict[str, object]) -> list[dict[str, object]]:
    items = payload.get("hits") or payload.get("documents") or payload.get("data") or []
    normalized = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        identifiers = item.get("identifiers") if isinstance(item.get("identifiers"), dict) else {}
        normalized.append({
            "title": item.get("title"), "doi": identifiers.get("doi") or item.get("doi"),
            "year": item.get("source", {}).get("publishYear") if isinstance(item.get("source"), dict) else item.get("year"),
            "journal": item.get("source", {}).get("sourceTitle") if isinstance(item.get("source"), dict) else item.get("journal"),
            "authors": item.get("names", {}).get("authors", []) if isinstance(item.get("names"), dict) else item.get("authors", []),
            "citation_count": item.get("citations", [{}])[0].get("count") if isinstance(item.get("citations"), list) and item.get("citations") else item.get("timesCited"),
            "source_ids": {"wos_uid": item.get("uid") or item.get("id")}, "sources": ["web_of_science"],
        })
    return [item for item in normalized if item.get("title")]


def _openalex_identity(item: dict[str, object]) -> dict[str, object]:
    summary = item.get("summary_stats") if isinstance(item.get("summary_stats"), dict) else {}
    institution = item.get("last_known_institutions") or ([item.get("last_known_institution")] if item.get("last_known_institution") else [])
    return {
        "name": item.get("display_name"), "source": "openalex",
        "identifiers": {"openalex_author_id": item.get("id"), "orcid": item.get("orcid")},
        "affiliations": [entry.get("display_name") for entry in institution if isinstance(entry, dict) and entry.get("display_name")],
        "document_count": item.get("works_count"), "citation_count": item.get("cited_by_count"), "h_index": summary.get("h_index"),
    }


def _openalex_work(work: dict[str, object]) -> dict[str, object]:
    location = work.get("primary_location") if isinstance(work.get("primary_location"), dict) else {}
    source = location.get("source") if isinstance(location.get("source"), dict) else {}
    return {
        "title": work.get("title"), "doi": work.get("doi"), "year": work.get("publication_year"),
        "journal": source.get("display_name"),
        "authors": [authorship.get("author", {}).get("display_name") for authorship in work.get("authorships", []) if isinstance(authorship, dict) and isinstance(authorship.get("author"), dict)],
        "citation_count": work.get("cited_by_count"), "source_ids": {"openalex_work_id": work.get("id")},
        "sources": ["openalex"],
    }


def _crossref_work(work: dict[str, object]) -> dict[str, object]:
    titles = work.get("title")
    published = work.get("published") if isinstance(work.get("published"), dict) else {}
    date_parts = published.get("date-parts", [])
    year = date_parts[0][0] if isinstance(date_parts, list) and date_parts and isinstance(date_parts[0], list) and date_parts[0] else None
    authors = [
        " ".join(part for part in (author.get("given"), author.get("family")) if isinstance(part, str))
        for author in work.get("author", []) if isinstance(author, dict)
    ]
    containers = work.get("container-title")
    return {
        "title": titles[0] if isinstance(titles, list) and titles else None, "doi": work.get("DOI"), "year": year,
        "journal": containers[0] if isinstance(containers, list) and containers else None, "authors": authors,
        "citation_count": work.get("is-referenced-by-count"), "source_ids": {"crossref_doi": work.get("DOI")},
        "sources": ["crossref"],
    }


def _normalize_name(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9가-힣]+", value.casefold()))


def _normalize_title(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9가-힣]+", value.casefold()))


def _name_similarity(left: str, right: str) -> float:
    left_terms = set(_normalize_name(left).split())
    right_terms = set(_normalize_name(right).split())
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / len(left_terms | right_terms)


def _to_int(value: object) -> int | None:
    try:
        return int(value) if value is not None and value != "" else None
    except (TypeError, ValueError):
        return None


def _year(value: object) -> int | None:
    if isinstance(value, str) and re.match(r"\d{4}", value):
        return int(value[:4])
    return _to_int(value)