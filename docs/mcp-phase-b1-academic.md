# MCP Phase B-1: Academic Intelligence

## Architecture

KIM/Qwen remains the planner. The existing multi-provider Academic Intelligence is exposed as one compact `academic` capability; provider-specific APIs and schemas are not added to the planning context. Detailed READ schemas are loaded only after Qwen selects the capability.

The semantic facade exposes exactly four tools:

- `academic_resolve_researcher`
- `academic_search_publications`
- `academic_get_researcher_evidence`
- `academic_compare_source_coverage`

Research actions use this MCP facade first. A direct compatibility fallback is allowed only when MCP execution did not begin, so one decision cannot duplicate paid or rate-limited provider calls.

## Provider Model

Curated identity and bibliometric sources are preferred when configured:

- Scopus
- Web of Science Researcher API and Starter documents, with independent entitlements

Public validation and fallback sources are:

- ORCID
- OpenAlex
- Semantic Scholar
- Crossref
- public Google Scholar profile discovery through ordinary Web search

Identity resolution and corpus aggregation remain separate. Names, aliases, affiliations, ORCID, Scopus Author ID, WoS Researcher ID, OpenAlex Author ID, and Semantic Scholar Author ID are retained as source-attributed evidence. Publications are deduplicated by DOI, then normalized title/year/authors. Citation and publication counts remain per-source rather than being summed into a false universal total.

Each provider reports one of `AVAILABLE_FULL`, `AVAILABLE_LIMITED`, `UNCONFIGURED`, `NO_ENTITLEMENT`, `RATE_LIMITED`, `UNAVAILABLE`, or `ERROR`. Missing credentials are therefore distinguishable from provider outages. On 2026-08-30 no Scopus or WoS credentials were configured; public-provider operation remained available with degraded health.

## Identity, Coverage, And Cache

A same-name match alone cannot establish high-confidence identity. Multiple same-source candidates produce `AMBIGUOUS`; contradictory cross-source affiliations lower confidence and mark a possible split profile. Unresolved candidates do not enter the verified merged corpus.

Provider-specific source records, retrieval timestamps, representative-paper selection reasons, coverage differences, and conflict assessments survive normalization. The six-hour cache key includes normalized name and affiliation hints to prevent namesake collisions. Transient provider failures use the shorter failure TTL.

## Dynamic Exposure And Overhead

The active planner sees one compact Academic capability description. Only after selection does it receive the four semantic schemas. Output from every facade tool is bounded to 12,000 characters.

Measured with the deployed Qwen tokenizer:

| Definition | Items | Tokens |
| --- | ---: | ---: |
| Academic detailed schemas | 4 tools | 325 |
| Active high-level catalog | 5 capabilities | 335 |

This avoids exposing Scopus, WoS, ORCID, OpenAlex, Semantic Scholar, and Crossref request schemas on every turn.

## A-F Benchmark

The production Qwen planner and deterministic identity fixtures produced these results:

| Case | Expected behavior | Result |
| --- | --- | --- |
| A: researcher capability evaluation | Discover public identity, then obtain scholarly identity/corpus/metrics | `scholarly=high`; Web discovery followed by `LOOKUP_AUTHOR`. Live public Academic evidence returned degraded/ambiguous rather than claiming a false match. |
| B: same-name researchers | Do not merge namesakes | `AMBIGUOUS`, two candidates, zero merged publications, identity and split-profile conflicts reported. |
| C: split/incomplete profiles | Preserve source differences and lower certainty | Scopus 120 versus OpenAlex 20 reported separately; affiliation mismatch and incomplete/split assessment reported; confidence lowered to `MEDIUM`. |
| D: current non-scholarly lookup | Use Web without Academic | `scholarly=low`; Web selected. |
| E: framework implementation question | Use Documentation/Web without Academic | Documentation and Web selected; Academic omitted. |
| F: papers announced today | Establish date and fresh primary sources before scholarly lookup | Production selector chose Time and Web. The research planner kept scholarly value high but continued discovery/fetch while evidence was snippet-level; it did not call Academic merely because the request mentioned papers. |

Case F is intentionally Web-first: same-day papers may not yet exist in scholarly indexes. Academic metadata becomes useful after fresh candidates are established and indexed; forcing it earlier would add latency without improving evidence.

A live public-provider smoke test for `안호선 / Ho Seon Ahn` returned ORCID, OpenAlex, Crossref, and limited Semantic Scholar states, while Scopus, WoS, and Google Scholar were `UNCONFIGURED`. It exposed four possible identity candidates, affiliation conflicts, OpenAlex split candidates, and publication-count discrepancy instead of merging records under a confident identity.

## Security And Credentials

The capability is READ-only. It exposes bounded normalized evidence, not raw provider responses. Credential-bearing exception text is not returned to the model or user. Credentials remain in ignored local environment configuration:

```dotenv
SCOPUS_API_KEY=
SCOPUS_INST_TOKEN=
WOS_API_KEY=
WOS_RESEARCHER_API_KEY=
```

No credential is required for the public fallback path. Provider unavailability degrades only the affected source and does not fail the entire research request.
