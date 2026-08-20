# Local Agent Web Chat

This LAN-only browser interface calls `runtime.AgentRuntime`, which loads the
existing agent policy and instruction files before calling the localhost vLLM
endpoint. It is not a direct browser-to-vLLM client.

Start the UI from the repository root:

```bash
./web/run.sh
```

The default listener is `0.0.0.0:8080`, while vLLM remains private at
`127.0.0.1:8000`. Open `http://117.16.245.72:8080` from a browser on a network
that permits this host and port. The UI requires a manually provisioned account;
there is no public registration. Create an initial administrator or guest from
the server terminal. The command prompts for the password so it is never placed
in shell history or the repository. Passwords must contain at least 8 characters:

```bash
/srv/local-ai-agent/venv/bin/python scripts/manage-users.py admin YOUR_ADMIN_USERNAME
/srv/local-ai-agent/venv/bin/python scripts/manage-users.py guest GUEST_USERNAME
```

Set `WEB_SESSION_SECRET` in the host `.env` before long-running use. The user
database is stored at `local-memory/web-users.sqlite3` and is ignored by Git.
The interface still uses HTTP by default. Do not expose it to the public
internet until it is placed behind HTTPS.

Endpoints:

- `GET /health`
- `POST /api/login`
- `POST /api/logout`
- `GET /api/agents`
- `POST /api/new-session`
- `POST /api/chat`

Sessions are process-memory only. A New Chat starts a fresh UUID session and
does not persist or alter long-term memory. Streaming is intentionally deferred
until the non-streaming runtime path is validated.