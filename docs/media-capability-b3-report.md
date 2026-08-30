# Media Capability Phase B-3 Report

Date: 2026-08-30

## Outcome

KIM now exposes a semantic Media capability over the existing authenticated image worker. `MediaDirector` is the canonical generation/edit/pose execution boundary shared by Web and MCP. It retains multi-intent operations, validates input and output images, normalizes worker/job/result status, keeps bytes behind opaque owner-scoped IDs, and optionally writes provenance-bearing Project artifacts.

No image service or Main GPU fallback was added.

## Acceptance Benchmarks

| Case | Scenario | Result | Evidence |
| --- | --- | --- | --- |
| A | Text-to-image | PASS | Metadata-only MCP result; live AHN7 generation returned a decodable 512x512 image. |
| B | Existing-image edit | PASS | Live AHN7 edit returned a decodable 512x512 image with the source logical ID retained. |
| C | Multi-intent edit | PASS | Portrait request planned `portrait.frontalize` before `image.edit`; Web and MCP share the same bounded retry pipeline. |
| D | Project artifact | PASS | Existing ProjectStore saved PNG output and JSON provenance with operation, worker, model, seed, source IDs, execution capabilities, and artifact creation time. |
| E | Worker offline/no fallback | PASS | Unconfigured registry returned `UNCONFIGURED`; local image endpoint was not called. Host preserves `BUSY` without retry. |
| F | No Media required | PASS | Media is opt-in and lazily exposed only when selected; compact catalog remains under 3000 characters. |
| G | Research plus image | PASS | Research decision protocol accepts `CREATE_MEDIA` only after evidence and can request optional current-Project storage. |
| H | Invalid/unauthorized/corrupt source | PASS | Opaque-ID validation, cross-project owner checks, confined Project reads, MIME/decode/byte/dimension checks, and malformed worker-output rejection are covered. |

## Live Validation

Authenticated calls used the existing Main loopback tunnel and bearer boundary.

- Generate: `SUCCEEDED`, 512x512, `image.generate` plus one bounded quality retry.
- Edit: `SUCCEEDED`, 512x512, `image.edit` plus one bounded completion retry.
- Worker after run: healthy, idle, zero failures.
- Main tunnel: `127.0.0.1:18010`.
- AHN7 worker: `127.0.0.1:8010`.
- Main image and pose services: disabled and inactive.

## Automated Validation

```text
python -m unittest discover -s tests
Ran 295 tests in 31.612s
OK
```

The previous baseline was 284 tests. Eleven focused Media tests were added for semantic discovery, byte isolation, session isolation, multi-intent ordering, no fallback, malformed output, OOM mapping, owner-scoped Project sources, artifact provenance, normalized host status, and Research composition.

## Rollback

Set `MCP_MEDIA_ENABLED=false` and restart the Web service. Existing Web image behavior continues to use the same Media runtime, while semantic MCP discovery and execution are disabled. The worker, tunnel, and Project storage formats require no rollback migration.
