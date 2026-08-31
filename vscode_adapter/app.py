"""Local OpenAI-compatible adapter for VS Code conversation compaction."""

from __future__ import annotations

import itertools
import json
import logging
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
UPSTREAM_BASE_URL = os.getenv("VSCODE_ADAPTER_UPSTREAM", "http://127.0.0.1:8000").rstrip("/")
SYNTHETIC_CONTINUATION = (
    "Continue the current task using the compacted conversation context and the available tool results."
)
SYNTHETIC_FINALIZATION = (
    "Provide the concise final answer now. Do not call tools and do not include analysis or private reasoning."
)
REQUEST_TIMEOUT = httpx.Timeout(180.0, connect=10.0)
TOKENIZER_PATH = os.getenv(
    "VSCODE_ADAPTER_TOKENIZER_PATH",
    "/srv/local-ai-agent/models/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
)
ADVERTISED_MAX_INPUT_TOKENS = int(os.getenv("VSCODE_ADAPTER_MAX_INPUT_TOKENS", "28672"))
BACKEND_MAX_MODEL_LEN = int(os.getenv("VSCODE_ADAPTER_MAX_MODEL_LEN", "32768"))
CLIENT_ID = "vscode"
DEFAULT_ROLE = "coder"
HOP_BY_HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer",
    "transfer-encoding", "upgrade", "host", "content-length",
}
RESPONSE_EXCLUDED_HEADERS = HOP_BY_HOP_HEADERS | {"content-encoding"}

logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)
request_ids = itertools.count(1)
metrics: dict[str, int | float] = {
    "requests_total": 0,
    "requests_2xx": 0,
    "requests_4xx": 0,
    "upstream_400": 0,
    "adapter_5xx": 0,
    "tool_tail_requests": 0,
    "synthetic_continuations": 0,
    "empty_upstream_finals": 0,
    "empty_output_repairs": 0,
    "final_text_success": 0,
    "missing_user_role": 0,
    "empty_user_query": 0,
    "synthetic_inserted": 0,
    "upstream_no_user_query_400": 0,
    "upstream_latency_ms_total": 0.0,
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.upstream = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
    try:
        yield
    finally:
        await app.state.upstream.aclose()


app = FastAPI(
    title="VS Code Qwen Compatibility Adapter",
    docs_url=None,
    redoc_url=None,
    lifespan=lifespan,
)


def _request_headers(request: Request) -> dict[str, str]:
    headers = {
        name: value for name, value in request.headers.items()
        if name.lower() not in HOP_BY_HOP_HEADERS
    }
    headers["x-ahnbys-client"] = CLIENT_ID
    headers["x-ahnbys-default-role"] = DEFAULT_ROLE
    return headers


def _response_headers(response: httpx.Response) -> dict[str, str]:
    return {
        name: value for name, value in response.headers.items()
        if name.lower() not in RESPONSE_EXCLUDED_HEADERS
    }


def _content_type(content: object) -> str:
    if content is None:
        return "null"
    if isinstance(content, str):
        return "string"
    if isinstance(content, list):
        return "array"
    if isinstance(content, dict):
        return "object"
    return type(content).__name__


def _meaningful_text_length(content: object) -> int:
    if isinstance(content, str):
        return len(content.strip())
    if not isinstance(content, list):
        return 0
    length = 0
    for part in content:
        if not isinstance(part, dict) or part.get("type") not in {"text", "input_text"}:
            continue
        text = part.get("text")
        if isinstance(text, str):
            length += len(text.strip())
    return length


@lru_cache(maxsize=1)
def _tokenizer():
    if not Path(TOKENIZER_PATH).is_dir():
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    except Exception:
        logger.exception("adapter_tokenizer_unavailable")
        return None


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") not in {"text", "input_text"}:
            continue
        text = part.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


def _token_count(value: str) -> int | None:
    tokenizer = _tokenizer()
    if tokenizer is None:
        return None
    return len(tokenizer.encode(value, add_special_tokens=False))


def _token_pressure(body: bytes, messages: list[object]) -> dict[str, object]:
    message_tokens = []
    role_tokens: dict[str, int] = {"system": 0, "user": 0, "assistant": 0, "tool": 0, "developer": 0}
    for message in messages:
        role = message.get("role") if isinstance(message, dict) else "invalid"
        count = _token_count(_content_text(message.get("content"))) if isinstance(message, dict) else 0
        message_tokens.append(count)
        if isinstance(count, int) and role in role_tokens:
            role_tokens[role] += count
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = {}
    tools = payload.get("tools") if isinstance(payload, dict) else None
    tool_schema_tokens = _token_count(json.dumps(tools, ensure_ascii=False, separators=(",", ":"))) if tools else 0
    raw_message_tokens = sum(count for count in message_tokens if isinstance(count, int))
    estimated_input_tokens = None
    tokenizer = _tokenizer()
    if tokenizer is not None and all(isinstance(message, dict) for message in messages):
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
            )
            estimated_input_tokens = _token_count(rendered)
        except Exception:
            estimated_input_tokens = raw_message_tokens + (tool_schema_tokens or 0)
    return {
        "message_content_tokens": message_tokens,
        "system_context_tokens": role_tokens["system"] + role_tokens["developer"],
        "user_context_tokens": role_tokens["user"],
        "assistant_context_tokens": role_tokens["assistant"],
        "tool_result_tokens": role_tokens["tool"],
        "tool_schema_tokens": tool_schema_tokens,
        "raw_content_plus_tools_tokens": raw_message_tokens + (tool_schema_tokens or 0),
        "estimated_input_tokens": estimated_input_tokens,
        "requested_max_tokens": payload.get("max_tokens") if isinstance(payload, dict) else None,
        "tool_count": len(tools) if isinstance(tools, list) else 0,
        "configured_vs_code_max_input_tokens": ADVERTISED_MAX_INPUT_TOKENS,
        "configured_vllm_max_model_len": BACKEND_MAX_MODEL_LEN,
    }


def has_meaningful_user_query(messages: list[object]) -> bool:
    return any(
        isinstance(message, dict)
        and message.get("role") == "user"
        and _meaningful_text_length(message.get("content")) > 0
        for message in messages
    )


def _message_structure(messages: list[object]) -> list[dict[str, object]]:
    structures = []
    for message in messages:
        if not isinstance(message, dict):
            structures.append({"role": "invalid", "content_type": "unavailable"})
            continue
        content = message.get("content")
        content_length = _meaningful_text_length(content)
        structures.append({
            "role": message.get("role") if isinstance(message.get("role"), str) else "invalid",
            "content_type": _content_type(content),
            "content_length": content_length,
            "content_empty": content_length == 0,
            "tool_calls": isinstance(message.get("tool_calls"), list) and bool(message["tool_calls"]),
            "tool_call_id": isinstance(message.get("tool_call_id"), str) and bool(message["tool_call_id"]),
            "name": isinstance(message.get("name"), str) and bool(message["name"]),
        })
    return structures


def _compatibility_payload(body: bytes) -> tuple[bytes, list[object], bool, bool, bool, bool]:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return body, [], False, False, False, False
    if not isinstance(payload, dict):
        return body, [], False, False, False, False
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return body, [], False, False, False, bool(payload.get("stream"))
    roles = [message.get("role") for message in messages if isinstance(message, dict)]
    valid_roles = all(isinstance(role, str) for role in roles) and len(roles) == len(messages)
    has_user = valid_roles and "user" in roles
    meaningful_user = valid_roles and has_meaningful_user_query(messages)
    stream = bool(payload.get("stream"))
    if has_user or not messages or not valid_roles:
        return body, messages, has_user, meaningful_user, False, stream
    has_assistant = "assistant" in roles
    has_tool = "tool" in roles
    continuation = has_assistant and has_tool and all(
        role in {"system", "user", "assistant", "tool", "developer"} for role in roles
    )
    if not continuation:
        return body, messages, has_user, meaningful_user, False, stream
    payload["messages"] = [
        *messages,
        {
            "role": "user",
            "content": SYNTHETIC_CONTINUATION,
        },
    ]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(), messages, has_user, meaningful_user, True, stream


def _record_request(
    request_id: int,
    body: bytes,
    messages: list[object],
    has_user: bool,
    meaningful_user: bool,
    synthetic: bool,
    stream: bool,
) -> None:
    metrics["requests_total"] += 1
    if not has_user:
        metrics["missing_user_role"] += 1
    if has_user and not meaningful_user:
        metrics["empty_user_query"] += 1
    if synthetic:
        metrics["synthetic_inserted"] += 1
        metrics["synthetic_continuations"] += 1
    structures = _message_structure(messages)
    tool_tail = bool(structures and structures[-1]["role"] == "tool")
    if tool_tail:
        metrics["tool_tail_requests"] += 1
    pressure = _token_pressure(body, messages)
    logger.info("adapter_request=%s", json.dumps({
        "request_id": request_id,
        "client": CLIENT_ID,
        "default_role": DEFAULT_ROLE,
        "message_count": len(messages),
        "ordered_roles": [item["role"] for item in structures],
        "messages": structures,
        "has_user_role": has_user,
        "has_meaningful_user_query": meaningful_user,
        "tool_tail": tool_tail,
        "synthetic_continuation": synthetic,
        "stream": stream,
        "token_pressure": pressure,
    }, separators=(",", ":")))


def _safe_upstream_error(content: bytes) -> dict[str, object] | None:
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return None
    return {
        "type": error.get("type"),
        "message": error.get("message"),
        "code": error.get("code"),
    }


def _record_upstream(
    request_id: int,
    status_code: int,
    started: float,
    content: bytes = b"",
    count_status: bool = True,
) -> None:
    latency_ms = round((perf_counter() - started) * 1000, 1)
    metrics["upstream_latency_ms_total"] += latency_ms
    if count_status and 200 <= status_code < 300:
        metrics["requests_2xx"] += 1
    if count_status and 400 <= status_code < 500:
        metrics["requests_4xx"] += 1
    if count_status and status_code == 400:
        metrics["upstream_400"] += 1
    if count_status and status_code >= 500:
        metrics["adapter_5xx"] += 1
    error = _safe_upstream_error(content) if status_code == 400 else None
    if count_status and error and "No user query found" in str(error.get("message")):
        metrics["upstream_no_user_query_400"] += 1
    logger.info("adapter_upstream=%s", json.dumps({
        "request_id": request_id,
        "upstream_status": status_code,
        "upstream_latency_ms": latency_ms,
        "error": error,
    }, separators=(",", ":")))


def _empty_output_diagnostics() -> dict[str, object]:
    return {
        "sse_event_count": 0,
        "text_delta_count": 0,
        "tool_call_delta_count": 0,
        "reasoning_delta_count": 0,
        "final_text_length": 0,
        "reasoning_length": 0,
        "tool_call_count": 0,
        "finish_reason": None,
        "done_sent": False,
        "repair_applied": False,
    }


def _observe_choice(diagnostics: dict[str, object], choice: object, *, streaming: bool) -> None:
    if not isinstance(choice, dict):
        return
    message = choice.get("delta") if streaming else choice.get("message")
    if not isinstance(message, dict):
        message = {}
    content = message.get("content")
    if isinstance(content, str) and content:
        diagnostics["text_delta_count"] = int(diagnostics["text_delta_count"]) + 1
        diagnostics["final_text_length"] = int(diagnostics["final_text_length"]) + len(content)
    reasoning = message.get("reasoning_content") or message.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        diagnostics["reasoning_delta_count"] = int(diagnostics["reasoning_delta_count"]) + 1
        diagnostics["reasoning_length"] = int(diagnostics["reasoning_length"]) + len(reasoning)
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        diagnostics["tool_call_delta_count"] = int(diagnostics["tool_call_delta_count"]) + 1
        diagnostics["tool_call_count"] = max(int(diagnostics["tool_call_count"]), len(tool_calls))
    finish_reason = choice.get("finish_reason")
    if isinstance(finish_reason, str):
        diagnostics["finish_reason"] = finish_reason


def _nonstream_diagnostics(content: bytes) -> dict[str, object]:
    diagnostics = _empty_output_diagnostics()
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return diagnostics
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if isinstance(choices, list):
        for choice in choices:
            _observe_choice(diagnostics, choice, streaming=False)
    return diagnostics


def _record_output(
    request_id: int,
    status_code: int,
    diagnostics: dict[str, object],
    phase: str = "upstream",
) -> None:
    has_text = int(diagnostics["final_text_length"]) > 0
    has_tools = int(diagnostics["tool_call_count"]) > 0
    if 200 <= status_code < 300 and has_text:
        metrics["final_text_success"] += 1
    if 200 <= status_code < 300 and not has_text and not has_tools:
        metrics["empty_upstream_finals"] += 1
    logger.info("adapter_output=%s", json.dumps({
        "request_id": request_id,
        "phase": phase,
        "upstream_status": status_code,
        **diagnostics,
    }, separators=(",", ":")))


def _finalization_payload(body: bytes, *, stream: bool) -> bytes | None:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
        return None
    payload["messages"] = [
        *payload["messages"],
        {"role": "user", "content": SYNTHETIC_FINALIZATION},
    ]
    payload.pop("tools", None)
    payload.pop("tool_choice", None)
    payload["max_tokens"] = 512
    payload["stream"] = stream
    chat_template_kwargs = payload.get("chat_template_kwargs")
    if not isinstance(chat_template_kwargs, dict):
        chat_template_kwargs = {}
    payload["chat_template_kwargs"] = {**chat_template_kwargs, "enable_thinking": False}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()


def _observe_sse_event(diagnostics: dict[str, object], event: bytes) -> None:
    for raw_line in event.splitlines():
        line = raw_line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if data == b"[DONE]":
            diagnostics["done_sent"] = True
            continue
        diagnostics["sse_event_count"] = int(diagnostics["sse_event_count"]) + 1
        try:
            payload = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        choices = payload.get("choices") if isinstance(payload, dict) else None
        if isinstance(choices, list):
            for choice in choices:
                _observe_choice(diagnostics, choice, streaming=True)


def _sse_diagnostics(content: bytes) -> dict[str, object]:
    diagnostics = _empty_output_diagnostics()
    for event in content.split(b"\n\n"):
        _observe_sse_event(diagnostics, event)
    return diagnostics


def _has_renderable_output(diagnostics: dict[str, object]) -> bool:
    return int(diagnostics["final_text_length"]) > 0 or int(diagnostics["tool_call_count"]) > 0


async def _observed_stream(
    response: httpx.Response,
    request_id: int,
    started: float,
    client: httpx.AsyncClient,
    headers: dict[str, str],
    body: bytes,
) -> AsyncIterator[bytes]:
    diagnostics = _empty_output_diagnostics()
    pending = b""
    held_events: list[bytes] = []
    renderable = False
    try:
        async for chunk in response.aiter_raw():
            pending += chunk
            while b"\n\n" in pending:
                event, pending = pending.split(b"\n\n", 1)
                framed_event = event + b"\n\n"
                _observe_sse_event(diagnostics, framed_event)
                if renderable:
                    yield framed_event
                    continue
                held_events.append(framed_event)
                if _has_renderable_output(diagnostics):
                    renderable = True
                    yield b"".join(held_events)
                    held_events.clear()
        if pending:
            _observe_sse_event(diagnostics, pending)
            if renderable:
                yield pending
            else:
                held_events.append(pending)
    finally:
        await response.aclose()
        _record_upstream(request_id, response.status_code, started)
    if renderable or not 200 <= response.status_code < 300:
        _record_output(request_id, response.status_code, diagnostics)
        if held_events:
            yield b"".join(held_events)
        return

    repair_body = _finalization_payload(body, stream=True)
    if repair_body is None:
        _record_output(request_id, response.status_code, diagnostics)
        yield b"".join(held_events)
        return

    diagnostics["repair_applied"] = True
    metrics["empty_output_repairs"] += 1
    _record_output(request_id, response.status_code, diagnostics)
    repair_started = perf_counter()
    repair_request = client.build_request(
        "POST",
        f"{UPSTREAM_BASE_URL}/v1/chat/completions",
        headers=headers,
        content=repair_body,
    )
    repair = await client.send(repair_request, stream=True)
    repair_content = await repair.aread()
    await repair.aclose()
    repair_diagnostics = _sse_diagnostics(repair_content)
    _record_upstream(request_id, repair.status_code, repair_started, count_status=False)
    _record_output(request_id, repair.status_code, repair_diagnostics, phase="finalization")
    if _has_renderable_output(repair_diagnostics):
        yield repair_content
    else:
        yield b"".join(held_events)


@app.get("/health")
async def health() -> dict[str, object]:
    total = int(metrics["requests_total"])
    latency_total = float(metrics["upstream_latency_ms_total"])
    return {
        "status": "ok",
        "upstream": UPSTREAM_BASE_URL,
        "client": CLIENT_ID,
        "default_role": DEFAULT_ROLE,
        "metrics": {
            **metrics,
            "upstream_latency_ms_average": round(latency_total / total, 1) if total else 0.0,
        },
    }


@app.get("/v1/models")
async def models(request: Request) -> Response:
    request_id = next(request_ids)
    started = perf_counter()
    response = await request.app.state.upstream.get(
        f"{UPSTREAM_BASE_URL}/v1/models",
        headers=_request_headers(request),
    )
    _record_upstream(request_id, response.status_code, started, count_status=False)
    return Response(response.content, status_code=response.status_code, headers=_response_headers(response))


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    request_id = next(request_ids)
    original_body = await request.body()
    body, messages, has_user, meaningful_user, synthetic, stream = _compatibility_payload(original_body)
    _record_request(request_id, original_body, messages, has_user, meaningful_user, synthetic, stream)
    started = perf_counter()
    upstream_request = request.app.state.upstream.build_request(
        "POST",
        f"{UPSTREAM_BASE_URL}/v1/chat/completions",
        headers=_request_headers(request),
        content=body,
    )
    upstream = await request.app.state.upstream.send(upstream_request, stream=True)
    headers = _response_headers(upstream)
    content_type = upstream.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        return StreamingResponse(
            _observed_stream(
                upstream,
                request_id,
                started,
                request.app.state.upstream,
                _request_headers(request),
                body,
            ),
            status_code=upstream.status_code,
            headers=headers,
        )
    content = await upstream.aread()
    await upstream.aclose()
    _record_upstream(request_id, upstream.status_code, started, content)
    diagnostics = _nonstream_diagnostics(content)
    if 200 <= upstream.status_code < 300 and not _has_renderable_output(diagnostics):
        repair_body = _finalization_payload(body, stream=False)
        if repair_body is not None:
            diagnostics["repair_applied"] = True
            metrics["empty_output_repairs"] += 1
            _record_output(request_id, upstream.status_code, diagnostics)
            repair_started = perf_counter()
            repair = await request.app.state.upstream.post(
                f"{UPSTREAM_BASE_URL}/v1/chat/completions",
                headers=_request_headers(request),
                content=repair_body,
            )
            repair_diagnostics = _nonstream_diagnostics(repair.content)
            _record_upstream(request_id, repair.status_code, repair_started, count_status=False)
            _record_output(request_id, repair.status_code, repair_diagnostics, phase="finalization")
            if _has_renderable_output(repair_diagnostics):
                return Response(
                    repair.content,
                    status_code=repair.status_code,
                    headers=_response_headers(repair),
                )
    else:
        _record_output(request_id, upstream.status_code, diagnostics)
    return Response(content, status_code=upstream.status_code, headers=headers)
