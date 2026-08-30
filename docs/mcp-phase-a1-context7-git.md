# MCP Phase A-1: Context7 and Git

## Architecture

KIM/Qwen remains the single planner. The existing capability registry gives Qwen a compact high-level catalog, then exposes detailed schemas only for selected capabilities. The existing MCP host performs discovery, timeout handling, health normalization, and structured execution. No new agent framework or MCP host was added.

Coding keeps host-native workspace read/search/edit and safe execution as defaults. `documentation` and `git` are optional preferred capabilities. Roles are expertise profiles, not capability prisons. Research retains its separate evidence planner and receives only Search/Fetch MCP tools; Documentation and Git are not injected into its tool catalog.

## Context7

Verified on 2026-08-30:

- official maintainer/repository: Upstash, `upstash/context7`
- repository state: active, not archived
- package: `@upstash/context7-mcp` 4.0.4
- license: MIT
- recommended connection: Streamable HTTP at `https://mcp.context7.com/mcp`
- credential: optional `CONTEXT7_API_KEY`; recommended for higher rate limits
- local facade transport: in-process MCP; it calls the official remote endpoint

Only two read-only facade tools are exposed:

- `resolve_library_id`
- `query_documentation`

Queries are bounded and reject credential-like text. Library IDs must match the Context7 `/org/project[/version]` form. Output is bounded to 12,000 characters. Empty successful responses become `DEGRADED`; upstream errors become `ERROR` or `RATE_LIMITED`. Host timeouts and connection failures remain non-fatal tool observations, allowing Qwen to continue from code and existing knowledge.

The security boundary is READ-only, but documentation queries leave the host. Do not put proprietary source, credentials, tokens, or secrets in a Context7 query. The API key is read from environment configuration and sent only as an authorization header; it is never model-visible.

## Git

Git uses six semantic read operations:

- `git_status`
- `git_diff`
- `git_log`
- `git_show`
- `git_blame`
- `git_branch_info`

Every command uses a fixed repository root. Paths must be relative, are resolved under that root, and cannot escape it. Revisions reject option injection. Output is bounded to 12,000 characters and commands time out after 10 seconds. No commit, push, checkout, reset, rebase, merge, branch deletion, shell, force push, or hard reset tool is registered. Unconditional shell-based Git reads were removed from the Web Coding path so semantic Git and generic execution are not both used for the same automatic inspection.

## Dynamic Exposure and Overhead

The capability selector sees compact metadata including provider, availability, health, cost class, READ permission, and relevance description. Host-native workspace operations are not optional selections. Qwen may select zero or more available capabilities. Detailed schemas are sent only after selection.

Measured with the deployed Qwen tokenizer and 32K runtime:

| Definition | Tools | Tokens |
| --- | ---: | ---: |
| Context7 documentation | 2 | 185 |
| Git | 6 | 413 |
| Context7 + Git | 8 | 596 |
| Active high-level capability catalog | 5 capabilities | 269 |

The combined detailed schemas are not present on requests that select neither capability.

## Live Benchmark

Planning and execution checks ran against deployed Qwen on 2026-08-30.

| Case | Selection | Result |
| --- | --- | --- |
| A: FastAPI current recommendation | `documentation` | Workspace found `FastAPI(lifespan=...)`; Context7 official docs confirmed lifespan is recommended and `on_event` is deprecated. No web/news search. |
| B: `SearchRouter.search()` history | `git` | `git_log`, `git_show`, `git_blame`, and `git_branch_info` returned repository-scoped history. No unrestricted shell tool was exposed. |
| C: dictionary vs list | none | Empty capability selection; no Context7, Git, or Search schema/call. |
| D: vLLM parser configuration | `documentation` | Local vLLM 0.27.1 uses `--enable-auto-tool-choice --tool-call-parser qwen3_coder`; current official documentation confirmed that pairing for Qwen tool calling. No academic/market search. |

The Research catalog regression check for an NVIDIA analysis contains Search/Fetch only and excludes Git and Context7. SearchRouter provider, circuit-breaker, fallback, cost, and security behavior is unchanged.