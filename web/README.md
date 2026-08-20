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
that permits this host and port. This interface is for internal testing only and
does not provide authentication, HTTPS, or internet exposure; do not expose it
to the public internet without adding an authentication boundary.

Endpoints:

- `GET /health`
- `GET /api/agents`
- `POST /api/new-session`
- `POST /api/chat`

Sessions are process-memory only. A New Chat starts a fresh UUID session and
does not persist or alter long-term memory. Streaming is intentionally deferred
until the non-streaming runtime path is validated.