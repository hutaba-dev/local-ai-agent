# VS Code Qwen Compatibility Adapter

## Purpose

VS Code agent tool calling works with the production Qwen endpoint. The compatibility issue occurs after conversation compaction: VS Code may send a continuation containing system, assistant, and tool messages but no user role, while the Qwen chat template requires a user query.

The adapter is a localhost-only compatibility boundary:

```text
VS Code -> 127.0.0.1:8001 -> adapter -> 127.0.0.1:8000 -> vLLM/Qwen
```

The production vLLM service, chat template, model ID, and `max_model_len=16384` remain unchanged.

## Policy

`POST /v1/chat/completions` is passed through without modification when a valid user role exists. All request fields, including tools, tool choice, sampling fields, stream options, stop sequences, and supported extensions, are preserved.

A synthetic user message is appended only when all conditions hold:

- `messages` is a non-empty, well-formed list;
- no user role exists;
- both assistant and tool roles exist;
- every role is system, developer, assistant, or tool.

The inserted text is:

> Continue the current task using the compacted conversation context and the available tool results.

It is appended after the tool tail so existing assistant `tool_calls` and tool-result adjacency is unchanged. It supplies no new task, inferred prompt, source text, or recovered user content. Empty or malformed requests are not repaired.

## Endpoints

- `GET /v1/models`: transparent upstream proxy
- `POST /v1/chat/completions`: transparent proxy with the narrow compatibility policy
- `GET /health`: adapter health and aggregate counters

SSE responses are streamed from upstream without collecting the complete response. Authorization is forwarded. Request bodies, prompts, tool results, source code, and credentials are never logged.

Set `VSCODE_ADAPTER_ROLE_DEBUG=true` temporarily to log only request ID, message count, role sequence, whether the request ends in a tool result, and whether insertion occurred. Synthetic insertions are logged even when debug logging is disabled.

## VS Code Configuration

Use this Custom Endpoint URL after the service is healthy:

```text
http://127.0.0.1:8001/v1/chat/completions
```

Keep the model ID as `qwen3.8-27b` and tool calling enabled.

## 16K Observation

The adapter fixes role compatibility, not context capacity. During real VS Code runs, record these content-free measurements from VS Code diagnostics and adapter logs:

- compaction count and timestamps per completed task;
- tool calls and substantive coding steps before each compaction;
- tool schema token estimate at the first model call;
- prompt tokens or context utilization when exposed by the client/upstream;
- message count and role sequence at adapter requests;
- total task completion quality and latency.

Do not change production context length as part of adapter rollout.

## 32K Maintenance Benchmark

If 16K causes frequent compaction or prevents useful coding depth, compare 16384 and 32768 in a maintenance window using the same model, prompt set, tool schemas, and concurrency. Measure peak VRAM per GPU, maximum stable concurrency, TTFT, decode tokens per second, completed agent quality, compaction frequency, tool-call success, and end-to-end latency. Warm each configuration before measurement, run repeated trials, and restore 16384 unless 32768 passes health, quality, and concurrency gates.

## Validation

Automated fixtures cover normal chat, normal user-plus-tool agent traffic, compacted user-less tool continuation, empty messages, SSE streaming, model discovery, header forwarding, and ordinary endpoint isolation. The production fixture demonstrated direct `8000` returning `No user query found in messages` while the same request through `8001` returned HTTP 200.
