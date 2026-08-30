# LLM Research Orchestration

## Design Principle

Qwen is the Research Planner and Orchestrator. Python executes validated actions and enforces safety and resource limits. Keyword-derived intent labels are observability metadata only; they do not grant, deny, or select tools.

The initial `ResearchPlan` is the single semantic decision object. It records search mode and depth, external-information need, freshness, evidence needs, primary/scholarly/market value, entities, unresolved questions, model-generated queries, preferred capabilities, source preferences, and answer readiness. The same plan is passed into every continuation iteration. Selecting the Research role changes expertise, not depth.

Each iteration supplies Qwen with the user goal, evidence and observations so far, unresolved work, live tool availability, cost and freshness characteristics, prior decisions, and remaining budgets. Qwen returns one structured next action:

- `SEARCH_WEB`
- `FETCH_PAGE`
- `SEARCH_ACADEMIC`
- `LOOKUP_AUTHOR`
- `SEARCH_DOCUMENT`
- `COMPARE_EVIDENCE`
- `ANALYZE`
- `CALCULATE`
- `FINAL_ANSWER`

There is no mandatory Search, Fetch, Analyze sequence.

## Tool Information

The planner sees concise descriptions and live availability for SearXNG, Serper, Brave, Secure Page Fetch, project document search, Scopus, Web of Science, OpenAlex, Semantic Scholar, Crossref, and ORCID. It also sees cost classes and freshness characteristics.

The policy is to use the least expensive tool likely to obtain sufficient evidence without sacrificing materially important evidence quality. Specialized scholarly sources are selected only when they can reduce uncertainty about the actual question.

## Executor Guardrails

The executor enforces:

- supported action and argument schemas
- provider availability
- URL and fetch security validation
- duplicate action suppression
- timeout and provider rate-limit handling
- maximum 12 iterations
- maximum 12 tool actions
- maximum 12 search queries
- scoped project-document access
- no pending critical work at finalization
- no progress message as a final answer

Budgets are upper bounds, not quotas.

Provider health, fallback, circuit breakers, quotas, rate limits, duplicate suppression, MCP availability, URL validation, secure page fetch, SSRF protection, timeouts, and execution budgets remain deterministic. These controls constrain how an action runs; they do not infer what the user means.

## Search Quality

Search quality reports relevance, authority, freshness, and spam risk separately. Relevance comes from query overlap and provider scores. Authority is a low-weight configurable prior from `infra/search-source-reputation.json`, not a source allowlist. Freshness requires dated results for current requests. Spam blocking remains deterministic. Result ranking preserves provider relevance and hostname diversity without company, topic, or domain-specific semantic rules.

## Failure Behavior

Invalid initial planner output does not invoke a second keyword classifier. Automatic routing falls back to `NO_SEARCH`; an explicitly selected Research role gets one generic `QUICK_SEARCH` query. Invalid continuation output gets one provider-neutral `auto` discovery search while budget remains, then finalizes with the available evidence.

## Finalization

`FINAL_ANSWER` is accepted only when `ready_to_answer=true` and no critical unresolved question remains. If final synthesis says that more searching or checking is needed, the executor records a failed finalization observation and returns control to Qwen for another next action. This fixes the generic failure where a correct decision to search more was previously returned to the user as the final response.

Simple and moderate work can use one direct synthesis call. Qwen requests the Analyst/Critic path for complex causal analysis, consequential market analysis, conflicting evidence, identity ambiguity, or scenario work. `FACT`, `INFERENCE`, `FORECAST`, and `UNKNOWN` remain available but are not a mandatory response template.

## Activity UI

Agent Activity displays observable decision summaries, actions, selected providers, queries, contextual evidence priorities, complexity, actual tools, and budget usage. It does not display private chain-of-thought.

## Regression Benchmark

Planning-only benchmark run on 2026-08-30 with deployed Qwen. It made seven model calls and zero search-provider calls. Mean planner latency was 7.5 seconds.

| Case | Before: deterministic semantic routing | After: Qwen plan |
| --- | --- | --- |
| NVIDIA recent issues | Company label; no reliable freshness match for “recent” | Deep, high freshness, web/news/market evidence |
| NVIDIA earnings impact on HBM vendors | Fixed NVIDIA IR/SEC/Reuters query bundle | Deep, high freshness, financial/news/supply-chain evidence |
| Professor Ahn Ho-seon's research capability | Professor regex enabled academic pipeline | Deep, identity disambiguation plus scholarly evidence |
| Latest vLLM Qwen tool parser usage | “latest” forced current-news expansion | Quick, official docs and GitHub preferred |
| Transformer attention principle | Generic/technical label if research ran | No search; ready to explain |
| Important AI papers announced today | Current plus academic regex gates | Quick current paper discovery, scholarly value normal |
| Unknown company and HBM impact | Uppercase `HBM` could be mistaken for the company entity | Deep, unknown company retained as entity and verification target |

Static semantic-routing symbol occurrences across the three runtime modules fell from 22 to 0. Hard-coded company domains and forced query expansions fell to 0. All seven after cases selected an appropriate depth; no provider or academic capability was executed during this planning benchmark. Unsupported-claim rate cannot be measured without executing search and synthesis, so it is covered separately by evidence-only synthesis tests.

Automated regression tests verify structured-plan parsing, invalid-output fallback, role/depth separation, provider-neutral fallback, explicit specialized-tool selection, and rejection of premature finalization.
