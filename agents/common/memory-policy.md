# Memory Policy

## Separation

Short-term conversation context is the active chat turn history and current
workspace/task state supplied to the model. It is session-scoped, may contain
temporary working details, and expires when the session ends.

Long-term memory is a user-approved record retained across sessions. It is not
the model context window and must be searched deliberately before relevant
records are added to a prompt. The initial implementation is an in-memory store
with no persistence. A future persistence adapter must store data outside this
repository in an ignored local state directory.

## Long-Term Data Model

Each `MemoryRecord` has these fields:

| Field | Meaning |
| --- | --- |
| `id` | Stable record identifier. |
| `kind` | `preference`, `project_fact`, `decision`, or `workflow`. |
| `content` | Concise, user-approved fact or instruction. |
| `tags` | Searchable, non-sensitive labels. |
| `created_at` | UTC creation timestamp. |
| `updated_at` | UTC last-update timestamp. |
| `source` | `user` or an approved agent action. |
| `expires_at` | Optional UTC expiration timestamp. |

## Storage Rules

- Save long-term memory only after an explicit user request or confirmation.
- Never store API keys, passwords, tokens, private keys, connection strings
  containing credentials, or other secrets in long-term memory.
- Keep records concise and factual. Do not store raw conversation transcripts,
  unverified inferences, or tool output by default.
- A record that is no longer true must be updated or deleted instead of leaving
  contradictory history.

## User Interface

The Main Agent recognizes these explicit operations and confirms their result:

| User intent | Operation |
| --- | --- |
| `Remember: <fact>` | Create a long-term memory record after secret screening. |
| `What do you remember about <query>?` | Search long-term records and return matching summaries and IDs. |
| `Forget memory <id>` | Delete that record and confirm deletion. |
| `Forget everything about <topic>` | Search, show the affected IDs, then request confirmation before deleting multiple records. |
| `Show my saved memories` | List concise records and IDs without exposing secrets. |

The user may always ask not to save information. Short-term context is not
promoted to long-term memory automatically.