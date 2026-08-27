# Conditional Multi-Provider Search

## Purpose

The Research Agent uses a quality-gated provider chain instead of calling Brave Search for every query:

1. Self-hosted SearXNG
2. Serper.dev, when `SERPER_API_KEY` is configured
3. Brave Search, using the existing `BRAVE_SEARCH_API_KEY`

Providers are called sequentially. The router stops as soon as the merged result set satisfies the quality gate. It does not fan out to every provider.

Academic Intelligence remains a separate source layer. Scopus, Web of Science, OpenAlex, Semantic Scholar, Crossref, and Unpaywall eligibility is still controlled by the original research intent plan.

## Architecture

`runtime/search_providers.py` owns the provider abstraction and conditional router. `runtime/web_search.py` keeps the existing synchronous `search()` and `search_many()` interfaces and adapts their callers to normalized results. `runtime/tool_registry.py` passes the source-plan intent, freshness, and primary-evidence requirements into the router.

The normalized result fields are:

- `title`
- `url`
- `snippet`
- `published_at`
- `source`
- `provider`
- `category`
- `rank`
- optional `engine` and `score`
- `providers_seen` for cross-provider provenance

URLs are canonicalized before deduplication. Fragments and common tracking parameters are removed, host casing and `www.` are normalized, and duplicate URLs retain all provider names in `providers_seen`.

## Quality-Gated Fallback

Quality evaluation considers:

- result count
- query-term relevance
- domain diversity
- trusted current-news domains
- official or primary sources
- publication/freshness signals
- spam-heavy result sets

Current-news searches require a recent or trusted signal. Requests whose source plan requires primary evidence continue until an official source is found. A primary-rich SearXNG set can stop even when noisy metasearch results lower the overall relevance ratio, provided at least two official and two relevant results exist and the set is not spam-heavy.

Fallback also occurs when a provider is unavailable, times out, returns HTTP 429 or 5xx, or has an open circuit breaker. An absent Serper key is reported as `UNCONFIGURED` and is skipped without failing the request.

## Query Cost Controls

Deep Research executes at most three initial queries by default. `INITIAL_SEARCH_QUERY_BUDGET` can set a value from 1 through 6; malformed values fall back to 3. Normalized query fingerprints prevent duplicates within one search batch and across follow-up rounds.

The in-process provider cache key contains provider, normalized query, category, and freshness. Default TTLs are:

- current/news: 1,200 seconds
- general web: 21,600 seconds
- academic intent: 259,200 seconds

Cache hits do not consume paid requests. Provider failures are counted by a circuit breaker; two consecutive failures open the provider for 120 seconds by default.

## Configuration

```dotenv
SEARXNG_URL=http://127.0.0.1:8088
SERPER_API_KEY=
BRAVE_SEARCH_API_KEY=...
SEARCH_PROVIDER_ORDER=searxng,serper,brave
SEARCH_PROVIDER_TIMEOUT_SECONDS=8
SEARCH_CACHE_ENABLED=true
SEARCH_CACHE_CURRENT_TTL_SECONDS=1200
SEARCH_CACHE_GENERAL_TTL_SECONDS=21600
SEARCH_CACHE_ACADEMIC_TTL_SECONDS=259200
SEARCH_CIRCUIT_FAILURE_THRESHOLD=2
SEARCH_CIRCUIT_COOLDOWN_SECONDS=120
INITIAL_SEARCH_QUERY_BUDGET=3
```

Keep real API keys only in the ignored host `.env` file. The existing Brave variable name is unchanged.

## SearXNG Deployment

The Compose definition is in `infra/searxng/compose.yaml`, with settings in `infra/searxng/settings.yml`.

```bash
docker compose -f infra/searxng/compose.yaml up -d
docker compose -f infra/searxng/compose.yaml ps
```

The host publishes only `127.0.0.1:8088`; SearXNG is not exposed through public interfaces. JSON output is enabled for the application adapter. Default SearXNG engines provide multiple ordinary web and news upstreams. No CAPTCHA bypass, anti-bot bypass, proxy rotation, or public ingress is configured.

Example smoke tests:

```bash
curl -fsS 'http://127.0.0.1:8088/search?q=NVIDIA&format=json&categories=web'
curl -fsS 'http://127.0.0.1:8088/search?q=OpenAI&format=json&categories=news'
```

## Observability

The Agent Activity panel reports:

- initial and follow-up query counts
- status and call count for SearXNG, Serper, and Brave
- success, failure, HTTP 429, and timeout counts
- cache hits and misses
- average provider latency
- fallback count and reasons

Tool details use `execution: conditional_fallback`. The Deep Research activity payload exposes the aggregate under `research.search`.

## Validation Matrix

Automated provider tests cover:

- A: sufficient SearXNG results produce zero Serper and Brave calls
- B: insufficient current-news results fall back only until quality is sufficient
- C: existing Academic Intelligence routing remains enabled only for academic or mixed intent
- D: unavailable SearXNG falls back to Serper
- E: unavailable SearXNG plus unconfigured Serper falls back to Brave
- F: Serper HTTP 429 falls back to Brave
- G: repeated searches use cache without another provider call
- H: duplicate URLs retain cross-provider provenance

The existing web/runtime suite also verifies page fetching, current-market source selection, academic routing, Deep Research follow-ups, Korean Naver/Reddit behavior, and browser access controls.

## Production Smoke Results

Validated on the local host with SearXNG bound to `127.0.0.1:8088`:

- container health: healthy
- web JSON: HTTP 200, 158 results, multiple engines
- news JSON: HTTP 200, 63 results from Bing News, DuckDuckGo News, Google News, Reuters, and Wikinews
- unavailable endpoint: `TEMPORARILY_UNAVAILABLE` with a bounded connection error
- Serper without a key: `UNCONFIGURED`
- NVIDIA primary-source query: SearXNG 1 call, paid providers 0 calls

## Five-Category Benchmark

The benchmark used one representative query per category with an empty in-process cache. Latency varies with upstream engines and should be treated as a sample, not an SLA.

| Category | Conditional results | Conditional paid calls | Conditional latency | Brave-only paid calls | Brave-only latency |
|---|---:|---:|---:|---:|---:|
| Current news | 8 | 1 | 2,100 ms | 1 | 570 ms |
| Financial/market | 8 | 0 | 1,415 ms | 1 | 843 ms |
| Company | 8 | 0 | 3,247 ms | 1 | 948 ms |
| Technical/general | 6 | 0 | 1,321 ms | 1 | 847 ms |
| Academic query | 8 | 0 | 1,786 ms | 1 | 820 ms |

Total paid search requests fell from 5 to 1, an 80% reduction. The current-news query alone required Brave verification because the SearXNG set did not satisfy the freshness/relevance gate. Academic Intelligence provider calls are not represented by this general-search paid-call count.

## Operations

Restart SearXNG independently from the web application:

```bash
docker compose -f infra/searxng/compose.yaml restart
```

Inspect health and recent logs:

```bash
docker compose -f infra/searxng/compose.yaml ps
docker compose -f infra/searxng/compose.yaml logs --tail=100 searxng
```

To temporarily force a provider during diagnosis, set `SEARCH_PROVIDER_ORDER` to a single provider and restart the web service. Restore `searxng,serper,brave` after the comparison.
