# VS Code Local Qwen Endpoint

## Custom Endpoint Values

Use these values in VS Code Stable when creating a Custom Endpoint:

| VS Code field | Value |
| --- | --- |
| Provider / API compatibility | OpenAI-compatible |
| API type | Chat Completions |
| Base URL | `http://127.0.0.1:8000/v1` |
| Model | `qwen3.8-27b` |
| API key | Leave unset. If the UI requires a value, use `<LOCAL_VLLM_API_KEY>`; this server does not validate it. |
| Tool calling | Supported. vLLM uses the `qwen3_coder` tool-call parser with automatic tool choice enabled. |

The server is intentionally bound only to `127.0.0.1`. Do not change the
default server binding to `0.0.0.0` for VS Code access.

## VS Code On Another PC

Run the tunnel on the PC where VS Code is running, then use the same Base URL
above in VS Code. The local bind on both sides keeps the API private:

```bash
ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -L 127.0.0.1:8000:127.0.0.1:8000 \
  root@KIM_SERVER
```

Use an SSH host alias instead of putting credentials in the command. Keep this
terminal running while VS Code uses the endpoint. SSH forwarding is the default
remote-access method; do not publish the vLLM port directly to the internet.

## Connection Prompts

After selecting the endpoint, use these prompts in order:

1. `In one sentence, explain what an OpenAI-compatible Chat Completions API is.`
2. `Read README.md in this repository and summarize the local API endpoint and its access boundary in three bullets.`
3. `Use the available file-listing tool to list the files under docs/, then report the filenames. Do not infer the list without calling a tool.`

The first prompt verifies basic chat, the second verifies repository context
access, and the third verifies an Agent-mode tool call. Tool availability is
controlled by the selected VS Code chat mode and workspace trust settings.