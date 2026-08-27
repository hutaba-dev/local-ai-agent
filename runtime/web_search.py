"""Bounded external web search for Research tasks."""

from __future__ import annotations

import os
import re
import ipaddress
import socket
import re
from time import perf_counter, sleep
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import httpx

from runtime.search_providers import SearchRequest, SearchRouter, normalize_query

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
BRAVE_IMAGE_ENDPOINT = "https://api.search.brave.com/res/v1/images/search"
NAVER_ENDPOINT = "https://openapi.naver.com/v1/search/webkr.json"
REDDIT_ENDPOINT = "https://www.reddit.com/search.json"
OPENALEX_WORKS_ENDPOINT = "https://api.openalex.org/works"
S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
UNPAYWALL_API_BASE = "https://api.unpaywall.org/v2"
KOREAN_PATTERN = re.compile(r"[\uac00-\ud7a3]")
USER_AGENT = "local-ai-agent-research/0.1"
MAX_SOURCE_COUNT = 5
MAX_SOURCE_CHARS = 6_000
MAX_SOURCE_HTML_CHARS = 1_000_000
MIN_SOURCE_CHARS = 200
S2_MAX_ATTEMPTS = 3


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self._parts.append(data)

    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self._parts)).strip()


@dataclass(frozen=True)
class SearchResult:
    provider: str
    title: str
    url: str
    description: str


def search(
    query: str,
    mode: str,
    context: dict[str, object] | None = None,
    include_metrics: bool = False,
) -> list[dict[str, object]] | tuple[list[dict[str, object]], dict[str, object]]:
    """Return normalized results through the conditional provider router."""
    result_count = 5 if mode == "QUICK_SEARCH" else 8
    results: list[dict[str, object]] = []
    errors: list[str] = []
    metrics: dict[str, object] = {}
    try:
        request = SearchRequest(
            query=query,
            category="news" if "CURRENT_NEWS" in set((context or {}).get("intents", [])) else "web",
            freshness=str((context or {}).get("freshness") or "") or None,
            count=result_count,
            language="ko" if KOREAN_PATTERN.search(query) else "en",
        )
        batch = SearchRouter().search(request, context)
        results.extend(batch.results)
        metrics = batch.metrics
    except RuntimeError as error:
        errors.append(str(error))
    if KOREAN_PATTERN.search(query) and _naver_configured():
        try:
            results.extend(_legacy_result(result) for result in _naver_search(query, result_count))
        except httpx.HTTPError as error:
            errors.append(str(error))
    if mode == "DEEP_RESEARCH":
        try:
            results.extend(_legacy_result(result) for result in _reddit_search(query, min(3, result_count)))
        except httpx.HTTPError:
            pass
    if not results:
        detail = "; ".join(errors) or "no configured search provider returned results"
        raise RuntimeError(f"web search is required but unavailable: {detail}")
    unique = _unique_dict_results(results)[:result_count]
    return (unique, metrics) if include_metrics else unique


def search_many(
    queries: tuple[str, ...],
    mode: str,
    context: dict[str, object] | None = None,
    include_metrics: bool = False,
) -> list[dict[str, object]] | tuple[list[dict[str, object]], dict[str, object]]:
    """Search deduplicated queries and preserve provider usage metrics."""
    deduplicated = tuple(dict.fromkeys(
        query.strip() for query in queries if query.strip() and normalize_query(query)
    ))
    batch = SearchRouter().search_many(deduplicated, mode, context)
    results = list(batch.results)
    if mode == "DEEP_RESEARCH":
        for query in deduplicated:
            try:
                results.extend(_legacy_result(result) for result in _reddit_search(query, 2))
            except httpx.HTTPError:
                continue
    unique = _unique_dict_results(results)[:24]
    return (unique, batch.metrics) if include_metrics else unique


def visual_search(query: str, result_count: int = 3) -> list[dict[str, str]]:
    """Return bounded Brave image references for private art-direction analysis."""
    if not os.getenv("BRAVE_SEARCH_API_KEY"):
        raise RuntimeError("configure BRAVE_SEARCH_API_KEY for visual reference search")
    response = httpx.get(
        BRAVE_IMAGE_ENDPOINT,
        headers={"Accept": "application/json", "X-Subscription-Token": os.environ["BRAVE_SEARCH_API_KEY"]},
        params={"q": query, "count": min(max(result_count, 1), 4), "safesearch": "strict"},
        timeout=12,
    )
    response.raise_for_status()
    references = []
    for item in response.json().get("results", [])[:result_count]:
        thumbnail = item.get("thumbnail") if isinstance(item, dict) else None
        thumbnail_url = thumbnail.get("src") if isinstance(thumbnail, dict) else None
        if isinstance(item, dict) and isinstance(item.get("title"), str) and isinstance(thumbnail_url, str):
            references.append({
                "title": item["title"][:200],
                "url": str(item.get("url", ""))[:1_000],
                "thumbnail_url": thumbnail_url[:2_000],
            })
    return references


def fetch_visual_thumbnails(references: list[dict[str, str]], limit: int = 3) -> tuple[tuple[str, str, bytes], ...]:
    """Fetch small public image-search thumbnails with SSRF and size bounds."""
    images = []
    for reference in references[:limit]:
        url = reference.get("thumbnail_url")
        if not isinstance(url, str):
            continue
        try:
            response, _final_url = _safe_fetch_binary(url)
            content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
            if content_type not in {"image/jpeg", "image/png", "image/webp"} or len(response.content) > 2_000_000:
                continue
            images.append((reference.get("title", "visual reference"), content_type, response.content))
        except (OSError, ValueError, httpx.HTTPError):
            continue
    return tuple(images)


def _query_terms(query: str) -> tuple[str, ...]:
    return tuple(term.lower() for term in re.findall(r"[\w가-힣]{2,}", query) if term.lower() not in {"교수", "대한", "연구", "평가", "근거", "최근", "논문"})


def _relevance_score(result: SearchResult, terms: tuple[str, ...]) -> int:
    haystack = f"{result.title} {result.description}".lower()
    return sum(1 for term in terms if term in haystack)


def _legacy_result(result: SearchResult) -> dict[str, object]:
    return {
        "title": result.title,
        "url": result.url,
        "snippet": result.description,
        "published_at": None,
        "source": urlparse(result.url).netloc.lower().removeprefix("www."),
        "provider": result.provider,
        "category": "web",
        "rank": 0,
        "engine": None,
        "score": None,
        "providers_seen": [result.provider],
    }


def _unique_dict_results(results: list[dict[str, object]]) -> list[dict[str, object]]:
    from runtime.search_providers import canonicalize_url

    seen: dict[str, int] = {}
    unique: list[dict[str, object]] = []
    for result in results:
        url = str(result.get("url", ""))
        canonical = canonicalize_url(url)
        if canonical in seen:
            current = unique[seen[canonical]]
            current["providers_seen"] = list(dict.fromkeys((
                *current.get("providers_seen", []), *result.get("providers_seen", []),
            )))
            continue
        seen[canonical] = len(unique)
        unique.append(result)
    return unique


def academic_papers(
    queries: tuple[str, ...],
    limit_per_query: int = 3,
    diagnostics: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Retrieve public, structured work metadata from OpenAlex for Deep Research."""
    papers: list[dict[str, object]] = []
    seen: set[str] = set()
    for query in queries:
        started = perf_counter()
        record: dict[str, object] = {"query": query, "success": False}
        try:
            response = httpx.get(
                OPENALEX_WORKS_ENDPOINT,
                params={"search": query, "per-page": limit_per_query, "select": "id,doi,title,publication_date,cited_by_count,authorships,primary_location"},
                headers={"User-Agent": USER_AGENT},
                timeout=12,
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            record["failure_reason"] = _http_failure_reason(error)
            record["duration_ms"] = round((perf_counter() - started) * 1000)
            if diagnostics is not None:
                diagnostics.append(record)
            continue
        works = response.json().get("results", [])
        record.update({
            "success": True,
            "duration_ms": round((perf_counter() - started) * 1000),
            "result_count": len(works) if isinstance(works, list) else 0,
        })
        if diagnostics is not None:
            diagnostics.append(record)
        for work in works:
            if not isinstance(work, dict) or not isinstance(work.get("title"), str):
                continue
            identifier = work.get("doi") or work.get("id")
            if not isinstance(identifier, str) or identifier in seen:
                continue
            seen.add(identifier)
            authorships = work.get("authorships", [])
            authors = [
                author["author"]["display_name"]
                for author in authorships
                if isinstance(author, dict) and isinstance(author.get("author"), dict)
                and isinstance(author["author"].get("display_name"), str)
            ][:12]
            location = work.get("primary_location")
            source = location.get("source") if isinstance(location, dict) else None
            papers.append({
                "title": work["title"],
                "doi": work.get("doi"),
                "openalex_url": work.get("id"),
                "publication_date": work.get("publication_date"),
                "cited_by_count": work.get("cited_by_count"),
                "authors": authors,
                "venue": source.get("display_name") if isinstance(source, dict) else None,
            })
    return papers[:12]


def s2_search_author(query: str) -> list[dict[str, object]]:
    """Return bounded Semantic Scholar author candidates as independent evidence."""
    payload = _s2_get("/author/search", {"query": query, "limit": 5, "fields": "name,paperCount,citationCount,hIndex,affiliations"})
    candidates = payload.get("data", [])
    return [
        {
            "author_id": candidate.get("authorId"),
            "name": candidate.get("name"),
            "paper_count": candidate.get("paperCount"),
            "citation_count": candidate.get("citationCount"),
            "h_index": candidate.get("hIndex"),
            "affiliations": candidate.get("affiliations", []),
        }
        for candidate in candidates
        if isinstance(candidate, dict) and isinstance(candidate.get("authorId"), str)
    ]


def s2_get_author(author_id: str) -> dict[str, object]:
    """Return one Semantic Scholar author record."""
    payload = _s2_get(f"/author/{author_id}", {"fields": "name,paperCount,citationCount,hIndex,affiliations"})
    return {
        "author_id": author_id,
        "name": payload.get("name"),
        "paper_count": payload.get("paperCount"),
        "citation_count": payload.get("citationCount"),
        "h_index": payload.get("hIndex"),
        "affiliations": payload.get("affiliations", []),
    }


def s2_get_author_papers(author_id: str) -> list[dict[str, object]]:
    """Return representative papers for a confirmed Semantic Scholar author."""
    payload = _s2_get(
        f"/author/{author_id}/papers",
        {"limit": 6, "fields": "title,year,citationCount,externalIds,venue,authors,abstract"},
    )
    return [_s2_paper_record(paper) for paper in payload.get("data", []) if isinstance(paper, dict)]


def s2_get_paper(paper_id: str) -> dict[str, object]:
    """Return detailed metadata for one Semantic Scholar paper."""
    payload = _s2_get(
        f"/paper/{paper_id}",
        {"fields": "title,year,citationCount,externalIds,venue,authors,abstract"},
    )
    return _s2_paper_record(payload)


def semantic_scholar_evidence(
    query: str,
    author_hints: tuple[str, ...] = (),
    diagnostics: list[dict[str, object]] | None = None,
) -> dict[str, object] | None:
    """Best-effort author and representative-paper cross-check for an evidence gap."""
    try:
        candidates: list[dict[str, object]] = []
        selected_query = query
        for candidate_query in (*_s2_author_queries(query), *author_hints):
            candidates = _s2_search_author(candidate_query, diagnostics)
            if candidates:
                selected_query = candidate_query
                break
        if not candidates:
            if diagnostics is not None:
                diagnostics.append({"operation": "author_resolution", "success": False, "failure_reason": "no_matching_author"})
            return None
        candidate = _select_s2_author(candidates, selected_query, query)
        if candidate is None:
            return None
        author_id = candidate["author_id"]
        if not isinstance(author_id, str):
            return None
        author = _s2_get_author(author_id, diagnostics)
        papers = _s2_get_author_papers(author_id, diagnostics)
        exact_name_matches = [
            candidate for candidate in candidates
            if isinstance(candidate.get("name"), str)
            and candidate["name"].casefold() == selected_query.casefold()
        ]
        return {
            "author": author,
            "representative_papers": papers,
            "identity_status": "ambiguous" if len(exact_name_matches) > 1 else "matched",
            "same_name_candidate_count": len(exact_name_matches),
        }
    except (RuntimeError, httpx.HTTPError, ValueError) as error:
        if diagnostics is not None:
            diagnostics.append({"operation": "semantic_scholar", "success": False, "failure_reason": _http_failure_reason(error)})
        return None


def unpaywall_get_oa_location(doi: str) -> dict[str, object] | None:
    """Locate legal open-access metadata for a DOI; never bypasses a paywall."""
    email = os.getenv("UNPAYWALL_EMAIL")
    if not email:
        return None
    normalized_doi = doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
    try:
        response = httpx.get(
            f"{UNPAYWALL_API_BASE}/{normalized_doi}",
            params={"email": email},
            headers={"User-Agent": USER_AGENT},
            timeout=12,
        )
        response.raise_for_status()
    except httpx.HTTPError:
        return None
    payload = response.json()
    location = payload.get("best_oa_location")
    if not isinstance(location, dict):
        return None
    url = location.get("url_for_pdf") or location.get("url")
    if not isinstance(url, str) or not url.startswith("https://"):
        return None
    return {
        "doi": payload.get("doi") or normalized_doi,
        "title": payload.get("title"),
        "oa_url": url,
        "host_type": location.get("host_type"),
        "license": location.get("license"),
        "version": location.get("version"),
    }


def unpaywall_oa_locations(papers: list[dict[str, object]], limit: int = 3) -> list[dict[str, object]]:
    """Find legal OA locations only for DOI-bearing representative-paper candidates."""
    locations = []
    for paper in papers:
        if len(locations) >= limit:
            break
        doi = paper.get("doi")
        if isinstance(doi, str):
            location = unpaywall_get_oa_location(doi)
            if location:
                locations.append(location)
    return locations


def _s2_get(path: str, params: dict[str, object], diagnostics: list[dict[str, object]] | None = None) -> dict[str, object]:
    headers = {"User-Agent": USER_AGENT}
    if api_key := os.getenv("S2_API_KEY"):
        headers["x-api-key"] = api_key
    last_error: httpx.HTTPError | None = None
    for attempt in range(S2_MAX_ATTEMPTS):
        started = perf_counter()
        try:
            response = httpx.get(f"{S2_API_BASE}{path}", params=params, headers=headers, timeout=12)
            status_code = getattr(response, "status_code", 200)
            if status_code == 429 or status_code >= 500:
                response.raise_for_status()
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Semantic Scholar returned an invalid response")
            if diagnostics is not None:
                diagnostics.append({"path": path, "attempt": attempt + 1, "success": True, "duration_ms": round((perf_counter() - started) * 1000)})
            return payload
        except httpx.HTTPStatusError as error:
            last_error = error
            if diagnostics is not None:
                diagnostics.append({"path": path, "attempt": attempt + 1, "success": False, "duration_ms": round((perf_counter() - started) * 1000), "failure_reason": _http_failure_reason(error)})
            if error.response.status_code != 429 and error.response.status_code < 500:
                raise
            if attempt < S2_MAX_ATTEMPTS - 1:
                sleep(2 ** attempt)
    if last_error:
        raise last_error
    raise RuntimeError("Semantic Scholar request failed")


def _s2_search_author(query: str, diagnostics: list[dict[str, object]] | None) -> list[dict[str, object]]:
    payload = _s2_get("/author/search", {"query": query, "limit": 5, "fields": "name,paperCount,citationCount,hIndex,affiliations"}, diagnostics)
    return [
        {
            "author_id": candidate.get("authorId"), "name": candidate.get("name"), "paper_count": candidate.get("paperCount"),
            "citation_count": candidate.get("citationCount"), "h_index": candidate.get("hIndex"), "affiliations": candidate.get("affiliations", []),
        }
        for candidate in payload.get("data", [])
        if isinstance(candidate, dict) and isinstance(candidate.get("authorId"), str)
    ]


def _s2_get_author(author_id: str, diagnostics: list[dict[str, object]] | None) -> dict[str, object]:
    payload = _s2_get(f"/author/{author_id}", {"fields": "name,paperCount,citationCount,hIndex,affiliations"}, diagnostics)
    return {"author_id": author_id, "name": payload.get("name"), "paper_count": payload.get("paperCount"), "citation_count": payload.get("citationCount"), "h_index": payload.get("hIndex"), "affiliations": payload.get("affiliations", [])}


def _s2_get_author_papers(author_id: str, diagnostics: list[dict[str, object]] | None) -> list[dict[str, object]]:
    payload = _s2_get(f"/author/{author_id}/papers", {"limit": 6, "fields": "title,year,citationCount,externalIds,venue,authors,abstract"}, diagnostics)
    return [_s2_paper_record(paper) for paper in payload.get("data", []) if isinstance(paper, dict)]


def _http_failure_reason(error: BaseException) -> str:
    if isinstance(error, httpx.TimeoutException):
        return "timeout"
    if isinstance(error, httpx.HTTPStatusError):
        return f"http_{error.response.status_code}"
    if isinstance(error, httpx.HTTPError):
        return "network_error"
    if isinstance(error, ValueError):
        return "parsing_error"
    return "request_error"


def _s2_author_queries(query: str) -> tuple[str, ...]:
    """Extract a bounded person-name query before falling back to the full text."""
    korean_match = re.search(r"([가-힣]{2,4})\s*교수", query)
    if korean_match:
        return (korean_match.group(1),)
    leading_romanized_name = re.match(
        r"\s*([A-Z][A-Za-z-]*\s+[A-Z][A-Za-z-]*)(?=[이가은는을를와과]\b)", query
    )
    if leading_romanized_name:
        return (leading_romanized_name.group(1).replace("-", " "),)
    match = re.search(
        r"([A-Z][A-Za-z-]*\s+[A-Z][A-Za-z-]*)\s+(?:professor|researcher|academic)\b",
        query,
        re.IGNORECASE,
    )
    if match:
        name = match.group(1).replace("-", " ")
        parts = name.split()
        reversed_given_name = " ".join((*reversed(parts[:-1]), parts[-1]))
        return tuple(dict.fromkeys((name, reversed_given_name)))
    words = query.split()
    candidates = [query]
    for length in (4, 3, 2):
        if len(words) >= length:
            candidates.append(" ".join(words[:length]))
    return tuple(dict.fromkeys(candidate for candidate in candidates if candidate.strip()))


def _select_s2_author(
    candidates: list[dict[str, object]],
    author_query: str,
    context_query: str,
) -> dict[str, object] | None:
    name_terms = set(_s2_name_terms(author_query))
    context_terms = set(_s2_name_terms(context_query))
    ranked = sorted(
        candidates,
        key=lambda candidate: _s2_candidate_score(candidate, name_terms, context_terms),
        reverse=True,
    )
    if not ranked or _s2_candidate_score(ranked[0], name_terms, context_terms) <= 0:
        return None
    return ranked[0]


def _s2_candidate_score(candidate: dict[str, object], name_terms: set[str], context_terms: set[str]) -> int:
    candidate_name = candidate.get("name")
    candidate_terms = set(_s2_name_terms(candidate_name)) if isinstance(candidate_name, str) else set()
    affiliations = candidate.get("affiliations")
    affiliation_terms = set(_s2_name_terms(" ".join(value for value in affiliations if isinstance(value, str)))) if isinstance(affiliations, list) else set()
    exact_name_bonus = 10 if candidate_terms == name_terms and name_terms else 0
    return exact_name_bonus + 3 * len(candidate_terms & name_terms) + len(affiliation_terms & context_terms)


def _s2_name_terms(value: str) -> tuple[str, ...]:
    return tuple(term.lower() for term in re.findall(r"[A-Za-z]+|[가-힣]+", value) if len(term) > 1)


def _s2_paper_record(paper: dict[str, object]) -> dict[str, object]:
    external_ids = paper.get("externalIds")
    doi = external_ids.get("DOI") if isinstance(external_ids, dict) else None
    return {
        "title": paper.get("title"),
        "year": paper.get("year"),
        "citation_count": paper.get("citationCount"),
        "doi": doi,
        "venue": paper.get("venue"),
        "authors": paper.get("authors", []),
        "abstract": paper.get("abstract"),
    }


def fetch_sources(
    results: list[dict[str, str]],
    limit: int = MAX_SOURCE_COUNT,
    include_metrics: bool = False,
) -> list[dict[str, str]] | tuple[list[dict[str, str]], list[dict[str, object]]]:
    """Fetch bounded text from public HTTPS result URLs for Deep Research.

    URLs are provider output but remain untrusted. Block private network targets,
    validate each redirect, and accept HTML only.
    """
    sources: list[dict[str, str]] = []
    fetches: list[dict[str, object]] = []
    for result in results:
        if len(sources) >= limit:
            break
        url = result.get("url")
        if not isinstance(url, str):
            continue
        started = perf_counter()
        measurement: dict[str, object] = {
            "url": url,
            "execution": "sequential",
            # httpx's high-level sync API does not expose TCP connect timing.
            "connect_time_ms": None,
            "success": False,
            "bytes": None,
            "text_length": 0,
        }
        try:
            response, final_url = _safe_fetch(url)
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" not in content_type:
                measurement["failure_reason"] = "non_html_content"
                continue
            extractor = _TextExtractor()
            extractor.feed(response.text[:MAX_SOURCE_HTML_CHARS])
            text = extractor.text()[:MAX_SOURCE_CHARS]
            if len(text) >= MIN_SOURCE_CHARS:
                sources.append({
                    "title": result.get("title", "Untitled source"),
                    "url": final_url,
                    "text": text,
                    "relevance_score": result.get("relevance_score", 0.4),
                })
                measurement.update({
                    "url": final_url,
                    "success": True,
                    "bytes": len(response.text.encode("utf-8")),
                    "text_length": len(text),
                })
            else:
                measurement["failure_reason"] = "insufficient_extracted_text"
        except (OSError, ValueError, httpx.HTTPError) as error:
            measurement["failure_reason"] = _fetch_failure_reason(error)
        finally:
            measurement["total_fetch_time_ms"] = round((perf_counter() - started) * 1000)
            fetches.append(measurement)
    return (sources, fetches) if include_metrics else sources


def _fetch_failure_reason(error: BaseException) -> str:
    if isinstance(error, httpx.TimeoutException):
        return "timeout"
    if isinstance(error, httpx.HTTPStatusError):
        return f"http_{error.response.status_code}"
    if isinstance(error, httpx.HTTPError):
        return "network_error"
    if isinstance(error, ValueError):
        return "validation_or_redirect_error"
    return "fetch_error"


def _safe_fetch(url: str) -> tuple[httpx.Response, str]:
    current_url = url
    for _ in range(4):
        _validate_public_https_url(current_url)
        response = httpx.get(
            current_url,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
            timeout=12,
            follow_redirects=False,
        )
        if response.is_redirect:
            location = response.headers.get("location")
            if not location:
                raise ValueError("redirect had no location")
            current_url = urljoin(current_url, location)
            continue
        response.raise_for_status()
        return response, current_url
    raise ValueError("too many redirects")


def _safe_fetch_binary(url: str) -> tuple[httpx.Response, str]:
    current_url = url
    for _ in range(4):
        _validate_public_https_url(current_url)
        response = httpx.get(
            current_url,
            headers={"User-Agent": USER_AGENT, "Accept": "image/jpeg,image/png,image/webp"},
            timeout=12,
            follow_redirects=False,
        )
        if response.is_redirect:
            location = response.headers.get("location")
            if not location:
                raise ValueError("redirect had no location")
            current_url = urljoin(current_url, location)
            continue
        response.raise_for_status()
        return response, current_url
    raise ValueError("too many redirects")


def _validate_public_https_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("only public HTTPS URLs are allowed")
    for address_info in socket.getaddrinfo(parsed.hostname, None, type=socket.SOCK_STREAM):
        address = ipaddress.ip_address(address_info[4][0])
        if not address.is_global:
            raise ValueError("non-public network targets are not allowed")


def _primary_providers(query: str):
    providers = []
    if KOREAN_PATTERN.search(query) and _naver_configured():
        providers.append(_naver_search)
    if os.getenv("BRAVE_SEARCH_API_KEY"):
        providers.append(_brave_search)
    if not providers and _naver_configured():
        providers.append(_naver_search)
    if not providers:
        raise RuntimeError("configure BRAVE_SEARCH_API_KEY or NAVER_SEARCH_CLIENT_ID and NAVER_SEARCH_CLIENT_SECRET")
    return providers


def _naver_configured() -> bool:
    return bool(os.getenv("NAVER_SEARCH_CLIENT_ID") and os.getenv("NAVER_SEARCH_CLIENT_SECRET"))


def _naver_search(query: str, result_count: int) -> list[SearchResult]:
    response = httpx.get(
        NAVER_ENDPOINT,
        headers={
            "X-Naver-Client-Id": os.environ["NAVER_SEARCH_CLIENT_ID"],
            "X-Naver-Client-Secret": os.environ["NAVER_SEARCH_CLIENT_SECRET"],
        },
        params={"query": query, "display": result_count, "sort": "sim"},
        timeout=12,
    )
    response.raise_for_status()
    return [
        SearchResult("naver", _strip_html(item.get("title", "")), item["link"], _strip_html(item.get("description", "")))
        for item in response.json().get("items", [])[:result_count]
        if isinstance(item, dict) and isinstance(item.get("link"), str)
    ]


def _brave_search(query: str, result_count: int) -> list[SearchResult]:
    response = httpx.get(
        BRAVE_ENDPOINT,
        headers={"Accept": "application/json", "X-Subscription-Token": os.environ["BRAVE_SEARCH_API_KEY"]},
        params={"q": query, "count": result_count, "safesearch": "moderate"},
        timeout=12,
    )
    response.raise_for_status()
    return [
        SearchResult("brave", item["title"], item["url"], item.get("description", ""))
        for item in response.json().get("web", {}).get("results", [])[:result_count]
        if isinstance(item, dict) and isinstance(item.get("title"), str) and isinstance(item.get("url"), str)
    ]


def _reddit_search(query: str, result_count: int) -> list[SearchResult]:
    response = httpx.get(
        REDDIT_ENDPOINT,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        params={"q": query, "limit": result_count, "sort": "relevance", "t": "year", "raw_json": 1},
        timeout=12,
    )
    response.raise_for_status()
    children = response.json().get("data", {}).get("children", [])
    return [
        SearchResult("reddit", item["data"]["title"], f"https://www.reddit.com{item['data']['permalink']}", item["data"].get("selftext", "")[:400])
        for item in children[:result_count]
        if isinstance(item, dict) and isinstance(item.get("data"), dict) and isinstance(item["data"].get("title"), str) and isinstance(item["data"].get("permalink"), str)
    ]


def _unique(results: list[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    unique_results = []
    for result in results:
        if result.url not in seen:
            seen.add(result.url)
            unique_results.append(result)
    return unique_results


def _strip_html(value: object) -> str:
    return re.sub(r"<[^>]+>", "", value) if isinstance(value, str) else ""