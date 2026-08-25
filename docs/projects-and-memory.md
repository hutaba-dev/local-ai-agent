# Persistent Projects and Long-Term Memory

## Architecture

KIM is the only project and memory authority. AHN7 remains an image compute worker and stores no project, conversation, file, artifact, or memory state.

```text
Browser
  -> KIM Web / AgentRuntime / Qwen TP2
     -> projects.db, FTS5 indexes (KIM NVMe)
     -> project files and artifacts (KIM LOCAL_AI_DATA HDD)
     -> AHN7 image worker when image compute is requested
        -> result returned to KIM and persisted as a project artifact
```

The remote image path is documented in [distributed-image-worker.md](distributed-image-worker.md). This project architecture does not change that worker, its tunnel, authentication, or concurrency.

## KIM Storage Layout

### Bulk data HDD

The primary WDC WUH721818ALE6L4 18 TB disk is mounted by UUID at `/srv/local-ai-data` as ext4.

```text
/srv/local-ai-data/
├── projects/<immutable-project-id>/
│   ├── files/
│   ├── artifacts/
│   ├── conversations/
│   └── archive/
├── artifacts/
├── conversations/
├── archives/
├── raw-files/
├── exports/
└── backups/
```

### Metadata NVMe

Latency-sensitive state remains on the system NVMe:

```text
/var/lib/local-ai-agent/
├── projects.db
├── projects.db-wal
├── projects.db-shm
├── indexes/
└── cache/
```

SQLite uses WAL mode, foreign keys, a busy timeout, and FTS5. Original PDFs, images, video, and documents are not duplicated into the NVMe database.

### Backup HDD

The second 18 TB HDD is mounted at `/srv/local-ai-backup`. `local-ai-project-backup.timer` creates a daily SQLite online backup under `metadata/`, verifies it with `PRAGMA integrity_check`, and keeps 14 copies.

**Primary HDD != Backup.** The metadata backup is not a full project-file mirror. Critical project documents still require an independent off-host backup policy.

Both HDD fstab entries use UUID, `nofail`, and a 10-second systemd device timeout so a missing HDD does not prevent KIM from booting.

## Project Model

A display name is metadata only. Filesystem paths use immutable random IDs such as `prj_<id>` and never a user-supplied project name.

SQLite tables:

- `projects`: owner, name, description, current summary, timestamps, archive state
- `conversations`: project, persistent title, timestamps
- `messages`: user and assistant final messages plus public tool metadata
- `files`: project-relative path, MIME type, size, SHA-256, index status, extracted text
- `artifacts`: creator, source message, description, and file reference
- `memories`: type, confidence, source, active state, and supersession link
- `memory_sources`: durable-memory provenance
- `project_events`: user-visible activity history
- `message_fts`, `memory_fts`, `file_chunk_fts`: SQLite FTS5 indexes

Private chain-of-thought, hidden reasoning, passwords, tokens, and secrets are not stored. Memory writes reject likely secrets.

## Conversations and Context

General Chat keeps its existing ephemeral session and temporary-upload behavior.

Project Chat persists every user message and assistant final answer. Multiple conversations share the same project summary, memories, and files. Restarting Web or KIM does not remove them.

The runtime assembles bounded context in this order:

1. Agent system instructions
2. Project metadata and current summary
3. Relevant active memories
4. Relevant FTS5 file excerpts
5. Recent messages from the selected conversation
6. Current user request and public tool results

The current vLLM context limit is 16,384 tokens. Project context is capped at 24,000 characters, with at most 12 memories, 6 file excerpts, and 10 recent messages. The runtime never scans the whole HDD for a request.

## Deep Research Integration

General Chat and Project Chat use the same `AgentRuntime` Deep Research engine. Project Chat adds bounded project context and scoped retrieval before the shared research pipeline; it does not use a reduced project-specific response generator.

Selecting the Research agent explicitly always enters this shared Deep Research engine. The semantic classifier still plans queries, but a `NO_SEARCH` or `QUICK_SEARCH` classification cannot downgrade an explicit Research request. Automatic routing remains classifier-driven. Bounded Project context is available to the classifier only for resolving references such as “this researcher”; it remains user context rather than external evidence.

Deep Research progresses through explicit planning, search, identity resolution, reading, verification, gap analysis, follow-up, and synthesis phases. Each evidence-gap decision is structured. When `ready_to_answer` is false, the runtime executes the proposed follow-up queries in another research round, up to four rounds. A person whose identity is unresolved after the first round receives at least one follow-up round.

Malformed or unavailable gap-analysis output is treated as an unresolved evidence gap, not as permission to answer early. The runtime generates fallback follow-up queries and continues within the same research budget.

For Korean person queries, the original Korean name is retained as an exact quoted search query. Romanized forms are optional additional queries and never replace the original name. Project summaries, memories, and file matches are marked as user workspace context rather than independent external evidence; claims about an external researcher must be supported by fetched web or academic sources.

Only terminal research output is returned to the user. Planner or progress text is rejected as a final answer and receives one terminal synthesis retry; if that retry is also progress text, the request fails instead of returning a successful intermediate response. Every successful Deep Research response executes Analyst, Critic, and Final Synthesis calls. Agent Activity reports the research mode, state history, rounds, queries, tools, source counts, entity confidence, gap status, termination reason, and whether final synthesis executed. Private reasoning is never exposed.

## Memory Lifecycle

After the assistant response is complete, a Starlette background task asks Qwen for strict structured output:

- durable memories only
- incremental project summary
- optional Markdown artifact request

Memory types are `fact`, `decision`, `goal`, `constraint`, `preference`, `todo`, `research_result`, and `summary`.

Exact duplicates are reused. A replacement decision can mark earlier entries inactive through `superseded_by`; history remains queryable. The Memory UI shows active and superseded entries and supports inline edit and two-step delete confirmation. A request not to remember content is part of the extraction policy.

Background extraction failure never changes the completed assistant response. Failures are logged to `local-ai-web.service` journald.

## Retrieval and Semantic Preparation

Initial retrieval uses SQLite FTS5 BM25 with owner and project filters across memory, file chunks, and conversation messages. Selected file chunks are read from the index; the HDD is accessed only when the original selected file is downloaded or read.

`semantic_search` and `hybrid_search` interfaces exist without installing a vector database. Future embeddings and vector indexes belong under the NVMe index path; raw content remains on the HDD.

## File Ingestion

Project-scoped uploads write the original bytes to the project HDD first and index extracted text in SQLite. General Chat uploads retain their existing temporary semantics.

Indexed formats include text, Markdown, JSON, CSV, PDF, DOCX, XLSX, PPTX, common source code/config files, and supported images/video through the existing extraction pipeline. Unsupported files are retained with `not_indexed` status.

Files record original name, project-relative path, MIME type, size, SHA-256, conversation source, index status, and creation time.

## Scoped Tools and Security

`runtime/project_tools.py` provides owner-scoped operations:

- project list/create/open
- file list/search/read/create/upload/move
- artifact save
- memory search/add/update
- conversation search
- semantic/hybrid search interfaces

No tool receives unrestricted access to `/srv/local-ai-data`. Absolute paths, `..`, path escape, and symlink traversal are rejected. Reads and writes re-check the authenticated owner and immutable project ID. File moves are limited to the project's `files/` and `artifacts/` trees.

Delete operations require an explicit UI action and a second confirmation click. Guest accounts cannot use persistent Projects.

## Web UI

The existing chat component is reused. The sidebar provides General Chat and nested Project conversations. A Project workspace has Chat, Files, Memory, and Activity views, with a mobile drawer at narrow widths.

Project requests include both `project_id` and `conversation_id`. Supplying only one is rejected. The Files view supports persistent upload, search/filter, download, and delete. Generated images and reports appear in the same file list as artifacts.

The storage indicator reports mount state, total bytes, and free bytes.

## Failure Behavior

### AHN7 offline

Project chat, memory, files, conversations, and non-image Qwen work remain available. Only image generation, image edit, and portrait pose correction return an image-worker unavailable error.

### Project HDD offline

Qwen, General Chat, Telegram, and the AHN7 worker remain available. Project status reports `STORAGE OFFLINE`. Project create, conversation/message writes, memory writes, files, artifacts, and Project Chat return HTTP 503. The application never falls back to the NVMe or another directory.

### KIM restart

The data and backup filesystems mount from UUID fstab entries. `local-ai-web.service` wants and starts after the data mount attempt, but does not require it. This preserves General Chat during a storage failure.

## Operations

```bash
findmnt /srv/local-ai-data /srv/local-ai-backup
df -hT /srv/local-ai-data /srv/local-ai-backup
curl -fsS http://127.0.0.1:7000/health
systemctl status local-ai-web.service local-ai-project-backup.timer
journalctl -u local-ai-web.service -n 100 --no-pager
systemctl start local-ai-project-backup.service
```

Recovery order:

1. Restore the primary HDD mount at `/srv/local-ai-data`.
2. Verify the expected UUID and ext4 filesystem.
3. Run `PRAGMA integrity_check` on `projects.db`.
4. If needed, restore the newest verified metadata backup while Web is stopped.
5. Verify project paths and restart Web.

Do not create a replacement project directory on the NVMe while the HDD is offline.
