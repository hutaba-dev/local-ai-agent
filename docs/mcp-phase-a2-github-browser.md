# MCP Phase A-2: GitHub and Browser

## Architecture

KIM/Qwen remains the planner. Qwen first selects compact capabilities and receives detailed READ schemas only for selected capabilities. The existing MCP host handles discovery, timeouts, health, and normalized observations. No rule forces GitHub or Browser calls.

Verified on 2026-08-30:

- official GitHub MCP: `github/github-mcp-server` 1.11.0, MIT, local stdio
- official Playwright MCP: `microsoft/playwright-mcp`, `@playwright/mcp` 0.0.79, Apache-2.0
- runtime: Node 22.21.0 and installed Chromium revision 1237
- GitHub 1.11.0 is outside the published affected ranges checked during the audit

## GitHub READ Boundary

The facade exposes six semantic READ tools:

- `github_search_code`
- `github_get_file`
- `github_read_commits`
- `github_read_issues`
- `github_get_pull_request`
- `github_read_releases`

The official child process runs with `--read-only` and an exact ten-tool upstream allowlist. Legacy or inherited `GITHUB_TOOLSETS` is removed before launch, so toolsets cannot broaden that list. The local facade is a second allowlist and bounds returned text to 12,000 characters. No create, update, merge, comment, label, branch, commit, push, or repository administration operation is registered.

Set `GITHUB_PERSONAL_ACCESS_TOKEN` to a fine-grained token with access only to required repositories and read-only Contents, Issues, Pull requests, and Metadata permissions. Without it, the capability remains `UNCONFIGURED` and no child process starts. Never put the token in prompts, source, or tracked configuration.

## Public Browser Boundary

The facade exposes four operations:

- `browse_page`
- `browse_click`
- `browse_type`
- `browse_select`

Each call starts an isolated, headless Chromium process with no saved or shared profile, blocks service workers, omits image responses, disables code generation, and closes the process after the call. Arbitrary Playwright code, JavaScript evaluation, file upload, downloads, screenshots, tracing, network mocking, and persistent sessions are not exposed.

Application URL validation and Playwright origin flags are defense in depth, not the SSRF boundary. Chromium is forced through a per-call localhost CONNECT proxy. The proxy permits port 443 only, resolves every destination itself, rejects the entire answer if any resolved address is non-global, connects to the validated address directly, and bounds per-connection and per-session bytes. Plain HTTP, loopback, private, link-local, metadata, multicast, reserved, and unspecified destinations are rejected. Redirects and subresources pass through the same proxy.

Browser remains unavailable unless both `MCP_PLAYWRIGHT_ENABLED=true` and `MCP_PLAYWRIGHT_EGRESS_GUARD=true` are set. Failures use host health values (`DEGRADED` or `ERROR`) and preserve a specific `failure_type` such as `TIMEOUT`, `BROWSER_CRASH`, or `NAVIGATION_FAILED`.

## Dynamic Exposure and Overhead

Detailed exposure remains capped at ten tools. Slots are allocated round-robin across selected capabilities so a mixed selection cannot starve one capability. GitHub plus Browser uses exactly all ten Phase A-2 tools.

Measured with the deployed Qwen tokenizer:

| Definition | Tools | Tokens |
| --- | ---: | ---: |
| GitHub | 6 | 744 |
| Browser | 4 | 519 |
| GitHub + Browser | 10 | 1,261 |
| Active high-level catalog | 7 available capabilities | 373 |

For an A-1+A-2 four-capability selection, the ten-tool bound remains active and round-robin allocation represents Documentation, Git, GitHub, and Browser rather than silently dropping a capability.

## Live Benchmark

The production Qwen selector ran with the exact deployed catalog and selector prompt:

| Case | Selection | Latency |
| --- | --- | ---: |
| A: remote GitHub release and issue | `github` | 349 ms |
| B: JavaScript-rendered public page | `browser` | 344 ms |
| C: remote release plus rendered demo | `github`, `browser` | 384 ms |
| D: local repository history | `git` | 343 ms |
| E: dictionary versus list | none | 313 ms |

The Browser end-to-end smoke test rendered `https://example.com` through the filtering proxy and found `Example Domain`. Official GitHub child discovery returned exactly the ten approved upstream tools. A live authenticated GitHub API read was not run because no real token was configured.