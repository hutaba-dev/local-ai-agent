# Agent Architecture

## Overview

The user normally talks to one Main Agent. The Main Agent performs general
conversation and secretary-style coordination, then delegates specialist work
only when appropriate. Qwen remains a separate local OpenAI-compatible model
backend; agent roles are client-side instruction and workflow bundles.

```mermaid
flowchart LR
    User --> Main[Main Agent]
    Main -->|workspace changes| Coding[Coding Agent]
    Main -. future evidence tasks .-> Research[Research Agent]
    Main -. future runtime tasks .-> Server[Server Agent]
    Main -->|Chat Completions| Qwen[Qwen vLLM API]
    Coding -->|Chat Completions| Qwen
```

Only Main and Coding are active roles. Research and Server are future roles,
not background processes or invented delegates.

## Role Boundaries

- Main Agent owns the user relationship, active-session context, memory
  operations, delegation decisions, and unified reporting.
- Coding Agent owns repository investigation, minimal edits, executable
  validation, failure repair, and diff evidence for assigned workspace work.
- All roles inherit the common constitution. Role-specific instructions add
  workflow but cannot relax common safety or secret-handling rules.

## Memory Architecture

Short-term context is an ephemeral active-session input to the model. It is not
saved automatically. Long-term memory is explicit, user-approved structured
data represented by `MemoryRecord` and managed by `InMemoryMemoryStore` for the
initial phase. The initial store is deliberately non-persistent.

Future persistence must be a small adapter writing outside Git to an ignored
local state path. It must preserve the memory policy's explicit-save,
search-by-query, delete-by-ID, multi-delete confirmation, and secret exclusion
rules.

## Delegation Lifecycle

1. Main confirms the user's goal and decides whether Coding specialization is needed.
2. Main sends the Coding Agent a complete typed handoff: goal, constraints,
   acceptance checks, and commit intent.
3. Coding Agent follows its inspect-edit-test-diff workflow and returns
   validation evidence plus a change summary.
4. Main gives the user one consolidated result, including remaining risk.