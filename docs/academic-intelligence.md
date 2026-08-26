# Multi-Source Academic Intelligence

Researcher evaluation uses `runtime/academic_intelligence.py` to resolve identity and reconstruct publication evidence before final synthesis. No provider's author record is treated as the complete career record by itself.

For Korean names, Deep Research first searches the exact native name. Qwen then derives plausible publication-name spellings from public profile evidence, while a standard Korean romanizer adds spacing variants. These aliases are candidates, not conclusions: Scopus affiliation, ORCID, and corpus size must confirm the selected Author ID. Institution or department text is never accepted as a person name.

## Source Roles

Identity and curated bibliometrics prefer:

1. Official university, lab, CV, and publication pages discovered by web search
2. ORCID public expanded search and identifiers exposed by provider records
3. Scopus Author records
4. Web of Science Researcher records

Publication and citation cross-checks use Scopus, Web of Science Starter, public Google Scholar profile discovery, OpenAlex, Semantic Scholar, Crossref, publisher pages, and institutional repositories. Google Scholar discovery uses normal Brave search results only. The runtime does not scrape Scholar pages, bypass CAPTCHA, or automate around access controls.

## Provider Access

The web service loads the ignored `.env` file from the repository root. It is `0600 root:root` on KIM. Add credentials there and restart `local-ai-web.service`; never commit them.

```dotenv
SCOPUS_API_KEY=
SCOPUS_INST_TOKEN=
WOS_API_KEY=
```

Scopus endpoints used:

- Author Search: `/content/search/author`
- Author Retrieval: `/content/author/author_id/{id}`
- Scopus Search: `/content/search/scopus`
- Abstract Retrieval: `/content/abstract/scopus_id/{id}` or `/content/abstract/doi/{doi}`
- Citation Overview: `/content/abstract/citations`

Elsevier API keys do not guarantee all views. Enhanced author data, abstracts, document retrieval, and citation views can depend on institutional entitlement. `SCOPUS_INST_TOKEN` is optional and must only contain a token issued by Elsevier.

Web of Science endpoints used:

- Researcher API: `https://api.clarivate.com/apis/wos-researcher/researchers`
- Starter documents: `https://api.clarivate.com/apis/wos-starter/v1/documents`

WoS Researcher API requires a paid license in addition to a Web of Science subscription. Starter API has separate plans; the free trial omits times-cited and is limited to 50 requests per day, while institutional plans can include citation counts and higher quotas.

On this server, no Scopus or WoS credentials were configured during implementation. Unauthenticated probes returned HTTP 401. The runtime therefore reports these sources as unavailable and continues with public sources.

## Source State

Each attempted source reports one of:

- `AVAILABLE_FULL`: requested profile/document fields were returned
- `AVAILABLE_LIMITED`: search or public metadata is available but profile, citation, or entitlement-dependent fields are incomplete
- `NO_ENTITLEMENT`: the provider returned HTTP 401 or 403
- `RATE_LIMITED`: the provider returned HTTP 429
- `UNAVAILABLE`: credential missing, network/parse failure, or no accessible public profile

Missing credentials and subscription limits never fail the entire Research request.

## Orchestration

Independent provider calls run concurrently. Selection is dynamic:

1. Configured Scopus and WoS sources run first with public Scholar profile discovery.
2. If Scopus and WoS both resolve an author and their document counts agree within 25%, public API fallback can be skipped for quota and latency control.
3. If a curated source is missing, limited, ambiguous, or conflicts with the other source, ORCID, OpenAlex, Semantic Scholar, and Crossref run concurrently as validation sources.
4. The aggregate identity, publication, and metric snapshot is cached by normalized researcher name for six hours. A rate-limited snapshot expires after 60 seconds so a temporary provider throttle does not poison later research.

The pipeline records:

```text
IDENTITY_RESOLUTION
AUTHOR_IDENTIFIER_RESOLUTION
MULTI_SOURCE_PUBLICATION_DISCOVERY
DEDUPLICATION
AUTHORSHIP_VERIFICATION
COVERAGE_CHECK
CITATION_METRIC_CROSS_CHECK
REPRESENTATIVE_PAPER_SELECTION
```

## Identity and Corpus Rules

The internal entity keeps ORCID, Scopus Author ID, Web of Science Researcher ID, OpenAlex Author ID, Semantic Scholar Author ID, and Google Scholar profile URL separately. Exact names alone cannot produce high confidence. Independent source agreement plus affiliation and identifier evidence is required.

Publication candidates are deduplicated in this order:

1. Exact normalized DOI
2. Normalized title, year, and author set

Single-source records under unresolved identity remain candidates with low authorship confidence. Only medium/high-confidence records enter `merged_verified_corpus` or representative-paper selection. Database counts are never summed.

Coverage and metrics remain source-specific. The orchestrator detects large publication-count, citation, h-index, and affiliation discrepancies. A much smaller record is marked `incomplete_split_or_misresolved_record`, which triggers further gap analysis rather than becoming the researcher's total output.

## Activity and Synthesis

Agent Activity reports provider state, identity sources/confidence, source-level publication candidates, coverage conflicts, merged verified corpus size, representative-paper count, and cache usage. It does not expose private reasoning.

Final synthesis receives only bounded structured evidence. It must explain database coverage differences, keep citation and h-index values attributed to their source, analyze representative works rather than metrics alone, and state when identity or corpus reconstruction remains incomplete.