# KIM Operations Runbook

## Canonical endpoints

| Surface | Endpoint | Expected state |
|---|---|---|
| vLLM | `http://127.0.0.1:8000/v1/models` | Qwen model listed |
| VS Code adapter | `http://127.0.0.1:8001/health` | HTTP 200 |
| Web internal | `http://127.0.0.1:7001/health` | HTTP 200 |
| Web TLS | nginx `:7000` | TLS UI/API reachable |
| AHN7 tunnel | `http://127.0.0.1:18010/health` | HTTP 200 and capability inventory |

Use the capability catalog as the canonical per-capability health source. Preserve `AVAILABLE`, `DEGRADED`, `UNCONFIGURED`, `UNAVAILABLE`, and `DISABLED`; do not collapse them to a boolean.

## First response

1. Confirm the affected client and retain its request/session identifier.
2. Check the owning endpoint, then the capability catalog, then the provider-specific state.
3. Check GPU process and memory state before restarting vLLM or Media.
4. Reproduce with one bounded semantic operation.
5. Record partial-provider failures and retries in the acceptance result.

## Recovery rules

- Empty synthesis: allow the runtime's single direct-synthesis retry. Repeated empty output is a model/adapter limitation, not permission to loop indefinitely.
- Research repetition: verify URL deduplication and stop at the round or 120-second guard. Return sourced partial findings.
- Academic degradation: inspect `provider_states`. A usable multi-provider response with individual 403/429 failures is `PASS_WITH_LIMITATION`.
- GitHub: no `GITHUB_PERSONAL_ACCESS_TOKEN` means `UNCONFIGURED`; do not substitute scraping.
- Browser/Fetch SSRF rejection: treat private-target rejection as expected security behavior.
- AHN7 502: inspect the returned operation error before changing GPU/toolchain settings. `No face detected` is an input/operation failure, not service or OOM failure.
- Media: use a valid portrait for pose acceptance. Check both main and AHN7 GPU memory and service health after the job.
- VS Code continuation: synthetic-user insertion may repair transport while final content remains empty. Report that as `PASS_WITH_LIMITATION` until non-empty completion is demonstrated.

## Validation order

Run behavior-focused unit tests with production MCP flags and provider credentials removed from the environment. Then run the full isolated suite. Live acceptance is separate: load production configuration, perform one bounded case per capability, and report external provider degradation honestly.

After a runtime deployment, restart only affected services and repeat endpoint, catalog, GPU, and one no-tool chat checks. Do not restart a healthy Media or vLLM service for a provider-level HTTP error.