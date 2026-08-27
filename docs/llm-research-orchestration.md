# LLM Research Orchestration

## Design Principle

Qwen is the Research Planner and Orchestrator. Python executes validated actions and enforces safety and resource limits. Keyword-derived intent labels are observability metadata only; they do not grant, deny, or select tools.

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

## Finalization

`FINAL_ANSWER` is accepted only when `ready_to_answer=true` and no critical unresolved question remains. If final synthesis says that more searching or checking is needed, the executor records a failed finalization observation and returns control to Qwen for another next action. This fixes the generic failure where a correct decision to search more was previously returned to the user as the final response.

Simple and moderate work can use one direct synthesis call. Qwen requests the Analyst/Critic path for complex causal analysis, consequential market analysis, conflicting evidence, identity ambiguity, or scenario work. `FACT`, `INFERENCE`, `FORECAST`, and `UNKNOWN` remain available but are not a mandatory response template.

## Activity UI

Agent Activity displays observable decision summaries, actions, selected providers, queries, contextual evidence priorities, complexity, actual tools, and budget usage. It does not display private chain-of-thought.

## Live Planner Benchmark

Run on 2026-08-27 with the deployed Qwen model. The benchmark invokes planning only and does not spend paid search quota.

| Case | Search decision | First next action | Contextual assessment |
| --- | --- | --- | --- |
| Nvidia recent issues | Quick research | SearXNG web search | freshness high; scholarly value low |
| Professor Ahn Ho-seon's research capability | Deep research | SearXNG identity/official-profile discovery | primary and scholarly value high; resolve identity before lookup |
| NVIDIA earnings impact on HBM vendors | Deep research | SearXNG web search across earnings, supply chain, and HBM fundamentals | freshness and primary importance high; scholarly value low; complex with critic |
| Why Transformers use attention | No search | None | stable conceptual explanation |
| Important AI papers announced today | Quick research | SearXNG current paper discovery | freshness and primary importance high; scholarly value normal, allowing later academic verification |

Automated regression tests additionally verify that an unready `FINAL_ANSWER` and a synthesis progress message both continue through the executor instead of reaching the user.
