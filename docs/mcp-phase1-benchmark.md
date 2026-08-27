# MCP Phase 1 Benchmark

Date: 2026-08-27

## Scope

This benchmark validates the Search/Fetch MCP migration without executing external search providers. It combines four live Qwen planner decisions with deterministic executor outcomes. Run it with:

```bash
/srv/local-ai-agent/venv/bin/python scripts/benchmark_mcp_phase1.py --iterations 100 --live-planner
```

## Qwen Planner Selection

The deployed `qwen3.8-27b` model saw the MCP-enabled capability catalog.

| Case | Expected | Selected | Provider / URL | Input tokens | Latency |
| --- | --- | --- | --- | ---: | ---: |
| Current NVIDIA news | `SEARCH_WEB` | `SEARCH_WEB` | `auto` | 4,420 | 4,094 ms |
| Researcher evaluation | `SEARCH_WEB` or `LOOKUP_AUTHOR` | `LOOKUP_AUTHOR` | `auto` | 4,431 | 4,280 ms |
| Sufficient fetched evidence | `FINAL_ANSWER` | `FINAL_ANSWER` | no additional call | 4,614 | 4,276 ms |
| Primary source selected | `FETCH_PAGE` | `FETCH_PAGE` | explicit `https://investor.nvidia.com/news` | 4,480 | 4,035 ms |

Selection accuracy was 4/4. The sufficient-evidence case made no unnecessary tool choice. The search case delegated provider policy to SearchRouter with `provider=auto`. The fetch case selected an explicit URL from the prior observation.

## Executor A-D Matrix

Each deterministic case ran 100 iterations. Median values measure Python dispatch overhead with mocked tool outcomes, not network or provider latency.

| Case | MCP attempts | MCP executions | Direct executions | Duplicate executions | Result | Median overhead |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| A: direct baseline | 0 | 0 | 100 | 0 | success | 0.016 ms |
| B: MCP success | 100 | 100 | 0 | 0 | success | 0.030 ms |
| C: unavailable before execution | 100 | 0 | 100 | 0 | success via fallback | 0.031 ms |
| D: timeout after execution began | 100 | 100 | 0 | 0 | bounded failure | 0.027 ms |

The benchmark generated zero paid API requests. Case C confirms one direct migration fallback only when MCP did not execute. Case D confirms uncertain post-invocation failure never produces a duplicate direct request.

## Context Overhead

| Catalog | Tools | Characters | Estimated tokens at 4 chars/token |
| --- | ---: | ---: | ---: |
| Direct | 12 | 3,240 | 810 |
| MCP Phase 1 | 11 | 3,513 | 879 |
| Delta | -1 | +273 | +69 |

The MCP host retains complete official SDK input schemas for discovery and validation. The Qwen planner receives a compact catalog without duplicated schema titles/defaults. The token value above is an estimate; live planner rows report the inference server's actual whole-prompt token counts.

## Regression Gates

Automated tests verify:

- official MCP discovery for `search_web`, `search_news`, and `fetch_page`
- SearchRouter reuse and fetch SSRF/security boundary reuse
- malformed input/result, tool-not-found, and server-down handling
- MCP feature flags and direct rollback
- no direct call after an MCP execution may have begun
- explicit fetch URLs and `provider=auto` decision parsing
- rejection of premature finalization and progress-only final output

Results establish the Phase 1 interface and failure semantics. They do not benchmark upstream search quality or network latency; those remain covered by the existing conditional multi-provider benchmark and production health metrics.

## Production Smoke

After enabling Phase 1, the web health endpoint passed and official-client discovery returned exactly `search_web`, `search_news`, and `fetch_page`. One live `provider=auto` search returned three results. Its quality gate used SearXNG and then one Brave request. A private-address fetch returned `ERROR` with no extracted text.
