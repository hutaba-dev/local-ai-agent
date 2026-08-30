"""Conditional multi-provider web search with normalized results and bounded cost."""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from time import monotonic, perf_counter
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

SEARXNG_DEFAULT_URL = "http://127.0.0.1:8088"
SERPER_SEARCH_ENDPOINT = "https://google.serper.dev/search"
SERPER_NEWS_ENDPOINT = "https://google.serper.dev/news"
BRAVE_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
USER_AGENT = "local-ai-agent-research/0.2"
TRACKING_PARAMETERS = {
    "fbclid", "gclid", "mc_cid", "mc_eid", "ref", "ref_src", "source",
}
REPUTATION_METADATA_PATH = Path(__file__).resolve().parents[1] / "infra" / "search-source-reputation.json"


def _load_reputation_metadata() -> dict[str, object]:
    try:
        value = json.loads(REPUTATION_METADATA_PATH.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


SOURCE_REPUTATION = _load_reputation_metadata()


class ProviderStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    UNCONFIGURED = "UNCONFIGURED"
    RATE_LIMITED = "RATE_LIMITED"
    TEMPORARILY_UNAVAILABLE = "TEMPORARILY_UNAVAILABLE"
    ERROR = "ERROR"


@dataclass(frozen=True)
class SearchRequest:
    query: str
    category: str = "web"
    freshness: str | None = None
    count: int = 10
    language: str | None = None


@dataclass(frozen=True)
class NormalizedSearchResult:
    title: str
    url: str
    snippet: str
    published_at: str | None
    source: str
    provider: str
    category: str
    rank: int
    engine: str | None = None
    score: float | None = None
    providers_seen: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        output = asdict(self)
        output["providers_seen"] = list(self.providers_seen or (self.provider,))
        return output


@dataclass(frozen=True)
class ProviderResponse:
    provider: str
    status: ProviderStatus
    results: tuple[NormalizedSearchResult, ...] = ()
    latency_ms: int = 0
    cache_hit: bool = False
    http_status: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class SearchQuality:
    sufficient: bool
    score: float
    reasons: tuple[str, ...]
    relevance_score: float = 0
    authority_score: float = 0
    freshness_score: float = 0
    spam_risk: float = 0


@dataclass(frozen=True)
class SearchBatch:
    results: tuple[dict[str, object], ...]
    metrics: dict[str, object]


@dataclass
class _CacheEntry:
    expires_at: float
    results: tuple[NormalizedSearchResult, ...]


class SearchCache:
    def __init__(self) -> None:
        self._entries: dict[tuple[str, str, str, str], _CacheEntry] = {}
        self._lock = Lock()

    def get(self, key: tuple[str, str, str, str]) -> tuple[NormalizedSearchResult, ...] | None:
        if not _env_bool("SEARCH_CACHE_ENABLED", True):
            return None
        now = monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at <= now:
                self._entries.pop(key, None)
                return None
            return entry.results

    def set(self, key: tuple[str, str, str, str], results: tuple[NormalizedSearchResult, ...], ttl: int) -> None:
        if not _env_bool("SEARCH_CACHE_ENABLED", True):
            return
        with self._lock:
            self._entries[key] = _CacheEntry(monotonic() + ttl, results)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


@dataclass
class _CircuitState:
    failures: int = 0
    open_until: float = 0


class CircuitBreaker:
    def __init__(self) -> None:
        self._states: dict[str, _CircuitState] = {}
        self._lock = Lock()

    def available(self, provider: str) -> bool:
        with self._lock:
            return self._states.get(provider, _CircuitState()).open_until <= monotonic()

    def success(self, provider: str) -> None:
        with self._lock:
            self._states[provider] = _CircuitState()

    def failure(self, provider: str) -> None:
        threshold = _env_int("SEARCH_CIRCUIT_FAILURE_THRESHOLD", 2, 1, 10)
        cooldown = _env_int("SEARCH_CIRCUIT_COOLDOWN_SECONDS", 120, 10, 3600)
        with self._lock:
            state = self._states.setdefault(provider, _CircuitState())
            state.failures += 1
            if state.failures >= threshold:
                state.open_until = monotonic() + cooldown

    def clear(self) -> None:
        with self._lock:
            self._states.clear()


SEARCH_CACHE = SearchCache()
CIRCUIT_BREAKER = CircuitBreaker()


class SearchProvider(ABC):
    name: str
    paid: bool = False

    @abstractmethod
    def configured(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def search(self, request: SearchRequest) -> ProviderResponse:
        raise NotImplementedError

    def _cached(self, request: SearchRequest) -> ProviderResponse | None:
        cached = SEARCH_CACHE.get(_cache_key(self.name, request))
        if cached is None:
            return None
        return ProviderResponse(self.name, ProviderStatus.AVAILABLE, cached, cache_hit=True)

    def _store(self, request: SearchRequest, results: tuple[NormalizedSearchResult, ...]) -> None:
        SEARCH_CACHE.set(_cache_key(self.name, request), results, _cache_ttl(request))


class SearXNGProvider(SearchProvider):
    name = "searxng"

    def __init__(self, url: str | None = None) -> None:
        self.url = (url or os.getenv("SEARXNG_URL", "")).rstrip("/")

    def configured(self) -> bool:
        return bool(self.url)

    def search(self, request: SearchRequest) -> ProviderResponse:
        if not self.configured():
            return ProviderResponse(self.name, ProviderStatus.UNCONFIGURED)
        cached = self._cached(request)
        if cached:
            return cached
        params: dict[str, object] = {
            "q": request.query, "format": "json", "categories": request.category,
            "language": request.language or "auto", "safesearch": 1,
        }
        time_range = _freshness_range(request.freshness)
        if time_range:
            params["time_range"] = time_range
        return self._request("GET", f"{self.url}/search", request, params=params)

    def _request(self, method: str, url: str, request: SearchRequest, **kwargs: object) -> ProviderResponse:
        started = perf_counter()
        try:
            response = httpx.request(method, url, timeout=_provider_timeout(), headers={"User-Agent": USER_AGENT}, **kwargs)
            response.raise_for_status()
            payload = response.json()
            raw_results = payload.get("results", []) if isinstance(payload, dict) else []
            results = tuple(
                NormalizedSearchResult(
                    title=str(item.get("title", ""))[:500],
                    url=str(item.get("url", ""))[:2000],
                    snippet=_clean_text(item.get("content"))[:1000],
                    published_at=_published_at(item.get("publishedDate")),
                    source=_source_name(item), provider=self.name, category=request.category, rank=index,
                    engine=str(item.get("engine")) if item.get("engine") else None,
                    score=float(item["score"]) if isinstance(item.get("score"), (int, float)) else None,
                    providers_seen=(self.name,),
                )
                for index, item in enumerate(raw_results[:request.count], 1)
                if isinstance(item, dict) and item.get("title") and item.get("url")
            )
            self._store(request, results)
            return ProviderResponse(self.name, ProviderStatus.AVAILABLE, results, _elapsed_ms(started))
        except httpx.TimeoutException:
            return ProviderResponse(self.name, ProviderStatus.TEMPORARILY_UNAVAILABLE, latency_ms=_elapsed_ms(started), error="timeout")
        except httpx.HTTPStatusError as error:
            status = ProviderStatus.RATE_LIMITED if error.response.status_code == 429 else (
                ProviderStatus.TEMPORARILY_UNAVAILABLE if error.response.status_code >= 500 else ProviderStatus.ERROR
            )
            return ProviderResponse(self.name, status, latency_ms=_elapsed_ms(started), http_status=error.response.status_code, error=f"http_{error.response.status_code}")
        except (httpx.HTTPError, ValueError) as error:
            return ProviderResponse(self.name, ProviderStatus.TEMPORARILY_UNAVAILABLE, latency_ms=_elapsed_ms(started), error=type(error).__name__)


class SerperProvider(SearchProvider):
    name = "serper"
    paid = True

    def configured(self) -> bool:
        return bool(os.getenv("SERPER_API_KEY"))

    def search(self, request: SearchRequest) -> ProviderResponse:
        if not self.configured():
            return ProviderResponse(self.name, ProviderStatus.UNCONFIGURED)
        cached = self._cached(request)
        if cached:
            return cached
        endpoint = SERPER_NEWS_ENDPOINT if request.category == "news" else SERPER_SEARCH_ENDPOINT
        payload: dict[str, object] = {"q": request.query, "num": request.count}
        if request.language:
            payload["hl"] = request.language
        if _freshness_range(request.freshness) == "day":
            payload["tbs"] = "qdr:d"
        started = perf_counter()
        try:
            response = httpx.post(
                endpoint,
                headers={"X-API-KEY": os.environ["SERPER_API_KEY"], "Content-Type": "application/json"},
                json=payload,
                timeout=_provider_timeout(),
            )
            response.raise_for_status()
            body = response.json()
            key = "news" if request.category == "news" else "organic"
            raw_results = body.get(key, []) if isinstance(body, dict) else []
            results = tuple(
                NormalizedSearchResult(
                    title=str(item.get("title", ""))[:500], url=str(item.get("link", ""))[:2000],
                    snippet=_clean_text(item.get("snippet"))[:1000], published_at=_published_at(item.get("date")),
                    source=str(item.get("source") or _host(item.get("link")))[:200], provider=self.name,
                    category=request.category, rank=index, providers_seen=(self.name,),
                )
                for index, item in enumerate(raw_results[:request.count], 1)
                if isinstance(item, dict) and item.get("title") and item.get("link")
            )
            self._store(request, results)
            return ProviderResponse(self.name, ProviderStatus.AVAILABLE, results, _elapsed_ms(started))
        except httpx.TimeoutException:
            return ProviderResponse(self.name, ProviderStatus.TEMPORARILY_UNAVAILABLE, latency_ms=_elapsed_ms(started), error="timeout")
        except httpx.HTTPStatusError as error:
            status = ProviderStatus.RATE_LIMITED if error.response.status_code == 429 else (
                ProviderStatus.TEMPORARILY_UNAVAILABLE if error.response.status_code >= 500 else ProviderStatus.ERROR
            )
            return ProviderResponse(self.name, status, latency_ms=_elapsed_ms(started), http_status=error.response.status_code, error=f"http_{error.response.status_code}")
        except (httpx.HTTPError, ValueError) as error:
            return ProviderResponse(self.name, ProviderStatus.TEMPORARILY_UNAVAILABLE, latency_ms=_elapsed_ms(started), error=type(error).__name__)


class BraveProvider(SearchProvider):
    name = "brave"
    paid = True

    def configured(self) -> bool:
        return bool(os.getenv("BRAVE_SEARCH_API_KEY"))

    def search(self, request: SearchRequest) -> ProviderResponse:
        if not self.configured():
            return ProviderResponse(self.name, ProviderStatus.UNCONFIGURED)
        cached = self._cached(request)
        if cached:
            return cached
        params: dict[str, object] = {
            "q": request.query, "count": request.count, "safesearch": "moderate",
            "search_lang": request.language or "en",
        }
        if request.category == "news" or _freshness_range(request.freshness) == "day":
            params["freshness"] = "pd"
        started = perf_counter()
        try:
            response = httpx.get(
                BRAVE_SEARCH_ENDPOINT,
                headers={"Accept": "application/json", "X-Subscription-Token": os.environ["BRAVE_SEARCH_API_KEY"]},
                params=params,
                timeout=_provider_timeout(),
            )
            response.raise_for_status()
            body = response.json()
            raw_results = body.get("web", {}).get("results", []) if isinstance(body, dict) else []
            results = tuple(
                NormalizedSearchResult(
                    title=str(item.get("title", ""))[:500], url=str(item.get("url", ""))[:2000],
                    snippet=_clean_text(item.get("description"))[:1000], published_at=_published_at(item.get("age")),
                    source=_host(item.get("url")), provider=self.name, category=request.category, rank=index,
                    providers_seen=(self.name,),
                )
                for index, item in enumerate(raw_results[:request.count], 1)
                if isinstance(item, dict) and item.get("title") and item.get("url")
            )
            self._store(request, results)
            return ProviderResponse(self.name, ProviderStatus.AVAILABLE, results, _elapsed_ms(started))
        except httpx.TimeoutException:
            return ProviderResponse(self.name, ProviderStatus.TEMPORARILY_UNAVAILABLE, latency_ms=_elapsed_ms(started), error="timeout")
        except httpx.HTTPStatusError as error:
            status = ProviderStatus.RATE_LIMITED if error.response.status_code == 429 else (
                ProviderStatus.TEMPORARILY_UNAVAILABLE if error.response.status_code >= 500 else ProviderStatus.ERROR
            )
            return ProviderResponse(self.name, status, latency_ms=_elapsed_ms(started), http_status=error.response.status_code, error=f"http_{error.response.status_code}")
        except (httpx.HTTPError, ValueError) as error:
            return ProviderResponse(self.name, ProviderStatus.TEMPORARILY_UNAVAILABLE, latency_ms=_elapsed_ms(started), error=type(error).__name__)


class SearchRouter:
    def __init__(self, providers: dict[str, SearchProvider] | None = None) -> None:
        self.providers = providers or {
            "searxng": SearXNGProvider(), "serper": SerperProvider(), "brave": BraveProvider(),
        }

    def search(self, request: SearchRequest, context: dict[str, object] | None = None) -> SearchBatch:
        context = context or {}
        merged: list[NormalizedSearchResult] = []
        responses: list[ProviderResponse] = []
        fallbacks: list[dict[str, str]] = []
        order = _provider_order()
        previous_quality = SearchQuality(False, 0, ("no_results",))
        for provider_name in order:
            provider = self.providers.get(provider_name)
            if provider is None:
                continue
            if not provider.configured():
                responses.append(ProviderResponse(provider_name, ProviderStatus.UNCONFIGURED))
                continue
            if not CIRCUIT_BREAKER.available(provider_name):
                responses.append(ProviderResponse(provider_name, ProviderStatus.TEMPORARILY_UNAVAILABLE, error="circuit_open"))
                continue
            response = provider.search(request)
            responses.append(response)
            if response.status == ProviderStatus.AVAILABLE:
                CIRCUIT_BREAKER.success(provider_name)
                merged = _merge_results(merged, list(response.results))
                previous_quality = evaluate_quality(merged, request, context)
                if previous_quality.sufficient:
                    break
                fallbacks.append({
                    "from": provider_name,
                    "reason": "; ".join(previous_quality.reasons) or "quality gate requested verification",
                })
            elif response.status not in {ProviderStatus.UNCONFIGURED}:
                CIRCUIT_BREAKER.failure(provider_name)
                fallbacks.append({"from": provider_name, "reason": response.error or response.status.value})
        if not merged:
            states = ", ".join(f"{item.provider}={item.status.value}" for item in responses) or "no providers"
            raise RuntimeError(f"web search is required but unavailable: {states}")
        return SearchBatch(
            tuple(item.to_dict() for item in merged[:request.count]),
            _metrics(order, responses, fallbacks, previous_quality),
        )

    def search_many(
        self,
        queries: tuple[str, ...],
        mode: str,
        context: dict[str, object] | None = None,
    ) -> SearchBatch:
        all_results: list[NormalizedSearchResult] = []
        reports: list[dict[str, object]] = []
        seen_queries: set[str] = set()
        for query in queries:
            fingerprint = normalize_query(query)
            if not fingerprint or fingerprint in seen_queries:
                continue
            seen_queries.add(fingerprint)
            category = _category_for(query, context or {})
            request = SearchRequest(
                query=query.strip(), category=category,
                freshness=str((context or {}).get("freshness") or "") or None,
                count=5 if mode == "QUICK_SEARCH" else 8,
                language="ko" if re.search(r"[가-힣]", query) else "en",
            )
            batch = self.search(request, context)
            reports.append(batch.metrics)
            all_results = _merge_results(
                all_results,
                [NormalizedSearchResult(**(item | {"providers_seen": tuple(item.get("providers_seen", []))})) for item in batch.results],
            )
        if not all_results:
            raise RuntimeError("no search results were returned")
        return SearchBatch(
            tuple(item.to_dict() for item in all_results[:24]),
            _aggregate_metrics(reports, len(seen_queries)),
        )


def evaluate_quality(
    results: list[NormalizedSearchResult], request: SearchRequest, context: dict[str, object]
) -> SearchQuality:
    if not results:
        return SearchQuality(False, 0, ("no results",))
    query_terms = set(re.findall(r"[\w가-힣]{3,}", normalize_query(request.query)))
    relevant = 0
    domains: set[str] = set()
    authority = 0.0
    primary = 0
    dated = 0
    spam = 0
    authority_domains = SOURCE_REPUTATION.get("authority_domains", {})
    authority_domains = authority_domains if isinstance(authority_domains, dict) else {}
    primary_markers = tuple(str(item) for item in SOURCE_REPUTATION.get("primary_url_markers", []) if isinstance(item, str))
    spam_markers = tuple(str(item) for item in SOURCE_REPUTATION.get("spam_block_markers", []) if isinstance(item, str))
    for result in results:
        text = f"{result.title} {result.snippet}".lower()
        overlap = len(query_terms & set(re.findall(r"[\w가-힣]{3,}", text)))
        relevant += int(not query_terms or overlap >= min(2, len(query_terms)))
        host = _host(result.url)
        domains.add(host)
        authority += max(
            (float(prior) for domain, prior in authority_domains.items() if str(domain) in host and isinstance(prior, (int, float))),
            default=0.0,
        )
        primary += int(any(marker in result.url.lower() for marker in primary_markers))
        dated += int(bool(result.published_at))
        spam += int(any(marker in f"{host} {text}" for marker in spam_markers))
    count_score = min(len(results) / 5, 1)
    relevance_score = relevant / len(results)
    diversity_score = min(len(domains) / 3, 1)
    authority_score = min((authority + primary) / len(results), 1)
    spam_ratio = spam / len(results)
    current = context.get("freshness") == "VERY_HIGH" or request.category == "news"
    freshness_score = dated / len(results) if current else 1
    score = max(0, 0.25 * count_score + 0.4 * relevance_score + 0.2 * diversity_score + 0.05 * authority_score + 0.1 * freshness_score - 0.3 * spam_ratio)
    reasons: list[str] = []
    if len(results) < 4:
        reasons.append("too few results")
    if relevance_score < 0.5:
        reasons.append("low query relevance")
    if len(domains) < min(3, len(results)):
        reasons.append("insufficient domain diversity")
    if current and not dated:
        reasons.append("no dated current result")
    if spam_ratio >= 0.5:
        reasons.append("spam-heavy result set")
    sufficient = score >= 0.62 and not reasons
    return SearchQuality(
        sufficient, round(score, 3), tuple(reasons), round(relevance_score, 3),
        round(authority_score, 3), round(freshness_score, 3), round(spam_ratio, 3),
    )


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip().casefold()


def canonicalize_url(url: str) -> str:
    try:
        parsed = urlsplit(url.strip())
        host = (parsed.hostname or "").lower().removeprefix("www.")
        port = f":{parsed.port}" if parsed.port and parsed.port not in {80, 443} else ""
        path = re.sub(r"/{2,}", "/", parsed.path or "/")
        if path != "/":
            path = path.rstrip("/")
        query = urlencode(sorted(
            (key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True)
            if not key.lower().startswith("utm_") and key.lower() not in TRACKING_PARAMETERS
        ))
    except ValueError:
        return url.strip()
    return urlunsplit((parsed.scheme.lower(), host + port, path, query, ""))


def reset_search_state() -> None:
    SEARCH_CACHE.clear()
    CIRCUIT_BREAKER.clear()


def _merge_results(existing: list[NormalizedSearchResult], incoming: list[NormalizedSearchResult]) -> list[NormalizedSearchResult]:
    merged = list(existing)
    indexes = {canonicalize_url(item.url): index for index, item in enumerate(merged)}
    for item in incoming:
        canonical = canonicalize_url(item.url)
        if canonical in indexes:
            index = indexes[canonical]
            current = merged[index]
            providers = tuple(dict.fromkeys((*current.providers_seen, current.provider, *item.providers_seen, item.provider)))
            merged[index] = NormalizedSearchResult(
                current.title, current.url, current.snippet or item.snippet,
                current.published_at or item.published_at, current.source or item.source,
                current.provider, current.category, current.rank, current.engine or item.engine,
                current.score if current.score is not None else item.score, providers,
            )
            continue
        indexes[canonical] = len(merged)
        merged.append(item)
    return merged


def _metrics(
    order: tuple[str, ...], responses: list[ProviderResponse], fallbacks: list[dict[str, str]], quality: SearchQuality
) -> dict[str, object]:
    stats: dict[str, dict[str, object]] = {}
    for response in responses:
        stats[response.provider] = {
            "status": response.status.value,
            "query_count": int(not response.cache_hit and response.status != ProviderStatus.UNCONFIGURED),
            "success_count": int(response.status == ProviderStatus.AVAILABLE),
            "failure_count": int(response.status not in {ProviderStatus.AVAILABLE, ProviderStatus.UNCONFIGURED}),
            "rate_limited_count": int(response.status == ProviderStatus.RATE_LIMITED),
            "timeout_count": int(response.error == "timeout"),
            "cache_hits": int(response.cache_hit),
            "average_latency_ms": response.latency_ms,
            "fallback_count": sum(1 for fallback in fallbacks if fallback.get("from") == response.provider),
        }
    return {
        "provider_order": list(order), "providers": stats, "fallbacks": fallbacks,
        "quality": asdict(quality),
        "cache_hits": sum(int(response.cache_hit) for response in responses),
        "cache_misses": sum(int(not response.cache_hit and response.status != ProviderStatus.UNCONFIGURED) for response in responses),
        "estimated_paid_requests": sum(
            int(not response.cache_hit and response.provider in {"serper", "brave"} and response.status != ProviderStatus.UNCONFIGURED)
            for response in responses
        ),
    }


def _aggregate_metrics(reports: list[dict[str, object]], query_count: int) -> dict[str, object]:
    providers: dict[str, dict[str, object]] = {}
    fallbacks: list[dict[str, str]] = []
    for report in reports:
        fallbacks.extend(report.get("fallbacks", []))
        for name, raw in report.get("providers", {}).items():
            item = raw if isinstance(raw, dict) else {}
            aggregate = providers.setdefault(name, {
                "status": item.get("status", ProviderStatus.UNCONFIGURED.value),
                "query_count": 0, "success_count": 0, "failure_count": 0,
                "rate_limited_count": 0, "timeout_count": 0, "cache_hits": 0,
                "total_latency_ms": 0, "fallback_count": 0,
            })
            aggregate["status"] = item.get("status", aggregate["status"])
            for key in ("query_count", "success_count", "failure_count", "rate_limited_count", "timeout_count", "cache_hits", "fallback_count"):
                aggregate[key] = int(aggregate[key]) + int(item.get(key, 0))
            aggregate["total_latency_ms"] = int(aggregate["total_latency_ms"]) + int(item.get("average_latency_ms", 0))
    for aggregate in providers.values():
        calls = int(aggregate["query_count"])
        aggregate["average_latency_ms"] = round(int(aggregate.pop("total_latency_ms")) / calls) if calls else 0
    return {
        "query_count": query_count,
        "provider_order": list(_provider_order()),
        "providers": providers,
        "fallbacks": fallbacks,
        "cache_hits": sum(int(report.get("cache_hits", 0)) for report in reports),
        "cache_misses": sum(int(report.get("cache_misses", 0)) for report in reports),
        "estimated_paid_requests": sum(int(report.get("estimated_paid_requests", 0)) for report in reports),
    }


def _provider_order() -> tuple[str, ...]:
    configured = os.getenv("SEARCH_PROVIDER_ORDER", "searxng,serper,brave")
    allowed = {"searxng", "serper", "brave"}
    order = tuple(dict.fromkeys(item.strip().lower() for item in configured.split(",") if item.strip().lower() in allowed))
    return order or ("searxng", "serper", "brave")


def _category_for(query: str, context: dict[str, object]) -> str:
    if "site:" in query.lower():
        return "web"
    category = context.get("search_category")
    return category if category in {"web", "news"} else "web"


def _cache_key(provider: str, request: SearchRequest) -> tuple[str, str, str, str]:
    return provider, normalize_query(request.query), request.category, (request.freshness or "").upper()


def _cache_ttl(request: SearchRequest) -> int:
    if request.freshness == "VERY_HIGH" or request.category == "news":
        return _env_int("SEARCH_CACHE_CURRENT_TTL_SECONDS", 1200, 60, 86400)
    if request.category == "academic":
        return _env_int("SEARCH_CACHE_ACADEMIC_TTL_SECONDS", 259200, 3600, 2592000)
    return _env_int("SEARCH_CACHE_GENERAL_TTL_SECONDS", 21600, 300, 604800)


def _freshness_range(freshness: str | None) -> str | None:
    return "day" if freshness == "VERY_HIGH" else None


def _provider_timeout() -> float:
    try:
        return min(max(float(os.getenv("SEARCH_PROVIDER_TIMEOUT_SECONDS", "8")), 1), 30)
    except ValueError:
        return 8


def _elapsed_ms(started: float) -> int:
    return round((perf_counter() - started) * 1000)


def _clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", str(value or ""))).strip()


def _published_at(value: object) -> str | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return str(value)[:100]


def _host(value: object) -> str:
    try:
        return (urlsplit(str(value or "")).hostname or "").lower().removeprefix("www.")
    except ValueError:
        return ""


def _source_name(item: dict[str, object]) -> str:
    return str(item.get("source") or _host(item.get("url")))[:200]


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        return min(max(int(os.getenv(name, str(default))), minimum), maximum)
    except ValueError:
        return default