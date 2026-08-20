"""Bounded external web search for Research tasks."""

from __future__ import annotations

import os
import re
import ipaddress
import socket
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import httpx

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
NAVER_ENDPOINT = "https://openapi.naver.com/v1/search/webkr.json"
REDDIT_ENDPOINT = "https://www.reddit.com/search.json"
OPENALEX_WORKS_ENDPOINT = "https://api.openalex.org/works"
S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
UNPAYWALL_API_BASE = "https://api.unpaywall.org/v2"
KOREAN_PATTERN = re.compile(r"[\uac00-\ud7a3]")
USER_AGENT = "local-ai-agent-research/0.1"
MAX_SOURCE_COUNT = 5
MAX_SOURCE_CHARS = 6_000


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


def search(query: str, mode: str) -> list[dict[str, str]]:
    """Return bounded search results chosen for Korean or global intent.

    Deep research may include Reddit discussion results as non-authoritative
    context. Provider failures are recorded only when no primary provider works.
    """
    result_count = 5 if mode == "QUICK_SEARCH" else 8
    providers = _primary_providers(query)
    results: list[SearchResult] = []
    errors: list[str] = []
    for provider in providers:
        try:
            results.extend(provider(query, result_count))
        except (RuntimeError, httpx.HTTPError) as error:
            errors.append(str(error))
    if mode == "DEEP_RESEARCH":
        try:
            results.extend(_reddit_search(query, min(3, result_count)))
        except httpx.HTTPError:
            pass
    if not results:
        detail = "; ".join(errors) or "no configured search provider returned results"
        raise RuntimeError(f"web search is required but unavailable: {detail}")
    return [asdict(result) for result in _unique(results)[:result_count]]


def search_many(queries: tuple[str, ...], mode: str) -> list[dict[str, str]]:
    """Search each bounded research query and deduplicate the combined results."""
    results: list[SearchResult] = []
    errors: list[str] = []
    for query in queries:
        try:
            results.extend(SearchResult(**result) for result in search(query, mode))
        except RuntimeError as error:
            errors.append(str(error))
    unique_results = _unique(results)
    anchor_terms = _query_terms(queries[0]) if queries else ()
    unique_results.sort(key=lambda result: _relevance_score(result, anchor_terms), reverse=True)
    unique = [asdict(result) for result in unique_results]
    if not unique:
        raise RuntimeError("; ".join(errors) or "no search results were returned")
    return unique[:24]


def _query_terms(query: str) -> tuple[str, ...]:
    return tuple(term.lower() for term in re.findall(r"[\w가-힣]{2,}", query) if term.lower() not in {"교수", "대한", "연구", "평가", "근거", "최근", "논문"})


def _relevance_score(result: SearchResult, terms: tuple[str, ...]) -> int:
    haystack = f"{result.title} {result.description}".lower()
    return sum(1 for term in terms if term in haystack)


def academic_papers(queries: tuple[str, ...], limit_per_query: int = 3) -> list[dict[str, object]]:
    """Retrieve public, structured work metadata from OpenAlex for Deep Research."""
    papers: list[dict[str, object]] = []
    seen: set[str] = set()
    for query in queries:
        try:
            response = httpx.get(
                OPENALEX_WORKS_ENDPOINT,
                params={"search": query, "per-page": limit_per_query, "select": "id,doi,title,publication_date,cited_by_count,authorships,primary_location"},
                headers={"User-Agent": USER_AGENT},
                timeout=12,
            )
            response.raise_for_status()
        except httpx.HTTPError:
            continue
        for work in response.json().get("results", []):
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


def semantic_scholar_evidence(query: str) -> dict[str, object] | None:
    """Best-effort author and representative-paper cross-check for an evidence gap."""
    try:
        candidates = s2_search_author(query)
        if not candidates:
            return None
        author = s2_get_author(candidates[0]["author_id"])
        papers = s2_get_author_papers(candidates[0]["author_id"])
        return {"author": author, "representative_papers": papers}
    except (RuntimeError, httpx.HTTPError, ValueError):
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


def _s2_get(path: str, params: dict[str, object]) -> dict[str, object]:
    headers = {"User-Agent": USER_AGENT}
    if api_key := os.getenv("S2_API_KEY"):
        headers["x-api-key"] = api_key
    response = httpx.get(f"{S2_API_BASE}{path}", params=params, headers=headers, timeout=12)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Semantic Scholar returned an invalid response")
    return payload


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


def fetch_sources(results: list[dict[str, str]], limit: int = MAX_SOURCE_COUNT) -> list[dict[str, str]]:
    """Fetch bounded text from public HTTPS result URLs for Deep Research.

    URLs are provider output but remain untrusted. Block private network targets,
    validate each redirect, and accept HTML only.
    """
    sources: list[dict[str, str]] = []
    for result in results:
        if len(sources) >= limit:
            break
        url = result.get("url")
        if not isinstance(url, str):
            continue
        try:
            response, final_url = _safe_fetch(url)
            content_type = response.headers.get("content-type", "").lower()
            if "text/html" not in content_type:
                continue
            extractor = _TextExtractor()
            extractor.feed(response.text[:MAX_SOURCE_CHARS * 3])
            text = extractor.text()[:MAX_SOURCE_CHARS]
            if text:
                sources.append({
                    "title": result.get("title", "Untitled source"),
                    "url": final_url,
                    "text": text,
                })
        except (OSError, ValueError, httpx.HTTPError):
            continue
    return sources


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