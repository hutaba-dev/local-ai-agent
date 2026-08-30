# Phase C Capability Acceptance

## Decision

Overall result: **PASS_WITH_LIMITATION**.

The integrated capability architecture is production-acceptable. Qwen owns auto-role and capability selection, permissions constrain tool exposure, selected and executed capabilities are observable, and bounded recovery prevents known synthesis/research loops. Remaining limitations are explicit below.

## Production scorecard

| Area | Result | Evidence |
|---|---|---|
| General no-tool chat | PASS | Completes without unnecessary capability calls |
| Auto role routing | PASS | Qwen planner role precedes deterministic fallback |
| Current news / Web | PASS_WITH_LIMITATION | Completes with four sources; external access and long synthesis remain variable |
| Academic | PASS_WITH_LIMITATION | Live evidence call succeeded with canonical `DEGRADED`; some providers returned 403/429 |
| Documentation | PASS | Context7 resolution and current documentation query completed |
| Coding / local Git | PASS | Read-only semantic inspection and scoped Coding permissions validated |
| GitHub | UNCONFIGURED | PAT intentionally absent; no scraping fallback |
| Project | PASS | Scoped context, memory, file, and artifact contracts covered by tests |
| Media generation | PASS | 512x512 generation completed on AHN7 |
| Media multi-intent edit | PASS | Valid portrait fixture completed pose adjustment then edit |
| Browser security | PASS | Private target rejected before navigation |
| Research recovery | PASS | URL deduplication, one-shot synthesis recovery, and elapsed guard validated |
| VS Code 32K transport | PASS_WITH_LIMITATION | Large request succeeds; synthetic continuation avoids transport error but can return empty final content |
| GPU/service stability | PASS | Main inference and AHN7 remained healthy after acceptance jobs |
| Telegram | PASS_WITH_LIMITATION | Existing client/service is operational; this phase did not introduce a new framework |

## Efficiency and observability

| Metric | Result |
|---|---:|
| Capabilities | 9 |
| Semantic tools | 40 |
| Detailed schema cap | 10 |
| Compact catalog | 549 tokens |
| Typical Web exposure | 226 tokens |
| Four-capability exposure | 934 tokens |
| Research elapsed guard | 120 seconds |

Activity reports the union of model-selected and actually executed capabilities. This keeps decisions visible even when the answer needs no tool call. Research call records expose duplicate suppression, failures, and retries.

## Failure and security acceptance

- Production feature flags are removed from unit-test runs to avoid ambient-state pollution.
- Role policies include all declared tool permission classes, including `EXECUTE_MEDIA` and scoped Project reads.
- Private Browser and Fetch destinations are blocked.
- Project access stays within the authorized project and opaque identifiers.
- Local Git and GitHub are read-only; destructive and repository-write permissions are excluded.
- Missing GitHub credentials remain `UNCONFIGURED`.
- Empty synthesis retries once, then terminates honestly.
- Academic partial-provider failures remain visible as `DEGRADED`.

## Known limitations

1. Current News can consume many rounds when source pages are inaccessible; guards bound this but cannot repair upstream access.
2. VS Code continuation transport is repaired, but Qwen may still emit an empty final assistant payload.
3. Academic provider quotas and anti-bot responses can reduce source coverage.
4. GitHub is unavailable until a read-only PAT is configured.
5. There is no standalone Market capability. Market questions intentionally route through existing Web/Research capabilities.

No new MCP server or agent framework was added in Phase C.