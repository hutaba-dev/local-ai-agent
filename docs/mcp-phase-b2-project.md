# MCP Phase B-2: Project Knowledge

## Decision

Phase B-2 wraps the existing AHNBYS Project system with a request-scoped semantic MCP facade. It does not create another memory store, vector database, or generic Filesystem MCP.

The sole persistence layer remains `runtime.projects.ProjectStore`:

- metadata and FTS: the existing Project SQLite database
- files and artifacts: the existing confined Project storage root
- durable memory: the existing typed memory and supersession tables
- provenance and audit: the existing artifact and project event tables

## Semantic tools

The Project capability exposes eight high-level operations only after an authenticated Project request selects the capability:

- `project_get_context`
- `project_search`
- `project_list_files`
- `project_read_file`
- `project_get_memories`
- `project_save_memory`
- `project_list_artifacts`
- `project_save_artifact`

Owner and Project IDs are injected by the host. Tool schemas never accept an owner, Project ID, or filesystem path. Files are read by opaque `fil_...` IDs. There are no delete, move, arbitrary path, shell, or generic filesystem operations.

## Runtime integration

Project Chat injects only bounded recent conversation context. Durable memories, files, excerpts, and artifact metadata are loaded lazily through Project MCP when they materially help.

Main and Coding use capability selection followed by detailed schema exposure. Deep Research reaches Project data only through the planner's `SEARCH_DOCUMENT` action. Direct and MCP retrieval are not both executed for the same Project path.

Memory writes are limited to durable, non-ephemeral knowledge. Artifact writes are limited to bounded text outputs requested for retention. Existing durable-update processing remains active after Project Chat responses.

## Security and failure behavior

- `ProjectStore._project_row` enforces owner and Project authorization.
- Project IDs and opaque file IDs are validated.
- Absolute paths, `..`, percent-encoded traversal, and direct or nested symlink escapes are rejected.
- Artifact MCP input accepts one filename, never a path.
- Cross-Project memory supersession is rejected.
- Raw `storage_path` values are removed from model-visible metadata.
- Missing Project storage returns `PROJECT_STORAGE_OFFLINE`; the MCP host treats it as unsuccessful and fail-closed.
- Identical artifact name/content writes are idempotent by SHA-256.

## Context and token budgets

Measured against the running `qwen3.8-27b` vLLM tokenizer:

| Payload | Characters | Tokens |
| --- | ---: | ---: |
| Compact Project capability catalog entry | 307 | 58 |
| Eight detailed Project tool schemas | 3,491 | 725 |
| Representative 4,000-character context | 4,000 | 2,000 |
| Maximum 12,000-character observation | 12,000 | 6,000 |

Detailed schemas are absent from general chat and appear only after Project capability selection. File reads are chunked with offsets and continuation metadata. Context retrieval accepts at most 12,000 characters; the normal request is 10,000.

## A-H verification

| Benchmark | Expected behavior | Evidence |
| --- | --- | --- |
| A. Memory recall | Relevant active memory is returned only from the scoped Project | scoped runtime/MCP tests pass |
| B. File retrieval | Opaque file ID returns bounded chunks and continuation offsets | 30,000-character chunk test passes |
| C. Project + Research/Web | Research planner can compose `SEARCH_DOCUMENT` with external actions | runtime path and live smoke |
| D. Coding + Project | Coding selects lazy Project schemas without duplicate preflight retrieval | tool-loop tests pass |
| E. General chat | Project catalog/schema/tools are unavailable without Project scope | negative lazy-exposure test passes |
| F. Unauthorized access | Cross-owner and cross-Project file/memory access fails | authorization tests pass |
| G. Path attack | absolute, traversal, encoded traversal, and symlink escapes fail | confinement tests pass |
| H. Storage offline | call executes fail-closed and is not reported available | host health test passes |

## Validation

Focused validation covers tool discovery, schemas, lazy exposure, authorization, cross-Project access, path confinement, large-file chunking, memory supersession, artifact idempotency, context bounds, offline health, and existing Project Chat behavior.

The complete repository suite passed: 284 tests.

Production Web/Qwen validation used signed temporary admin sessions and isolated temporary Projects against the running service and `qwen3.8-27b`:

- Main memory/file retrieval returned HTTP 200, selected the Project capability, called `project_search`, `project_list_files`, and `project_read_file`, and reproduced both synthetic values exactly.
- Deep Research returned HTTP 200, executed `SEARCH_DOCUMENT`, `SEARCH_WEB`, `FETCH_PAGE`, and `FINAL_ANSWER`, used successful Project and Web observations, reproduced both synthetic Project values, and completed final synthesis.
- The running Web health endpoint recovered after deployment.
- All temporary benchmark Projects were deleted; the production database reported zero remaining Phase B-2 benchmark Projects.
