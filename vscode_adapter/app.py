"""Local OpenAI-compatible adapter for VS Code conversation compaction."""

from __future__ import annotations

import itertools
import json
import logging
import os
from contextlib import asynccontextmanager
from time import perf_counter
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from starlette.background import BackgroundTask


UPSTREAM_BASE_URL = os.getenv("VSCODE_ADAPTER_UPSTREAM", "http://127.0.0.1:8000").rstrip("/")
ROLE_DEBUG = os.getenv("VSCODE_ADAPTER_ROLE_DEBUG", "false").strip().lower() in {"1", "true", "yes", "on"}
SYNTHETIC_CONTINUATION = (
    "Continue the current task using the compacted conversation context and the available tool results."
)
REQUEST_TIMEOUT = httpx.Timeout(180.0, connect=10.0)
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
    "requests_with_user_role": 0,
    "requests_missing_user_role": 0,
    "synthetic_continuation_inserted": 0,
    "upstream_400": 0,
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
    return {
        name: value for name, value in request.headers.items()
        if name.lower() not in HOP_BY_HOP_HEADERS
    }


def _response_headers(response: httpx.Response) -> dict[str, str]:
    return {
        name: value for name, value in response.headers.items()
        if name.lower() not in RESPONSE_EXCLUDED_HEADERS
    }


def _compatibility_payload(body: bytes) -> tuple[bytes, list[str], bool, bool]:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return body, [], False, False
    if not isinstance(payload, dict):
        return body, [], False, False
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return body, [], False, False
    roles = [message.get("role") for message in messages if isinstance(message, dict)]
    valid_roles = all(isinstance(role, str) for role in roles) and len(roles) == len(messages)
    has_user = valid_roles and "user" in roles
    if has_user or not messages or not valid_roles:
        return body, roles if valid_roles else [], has_user, False
    has_assistant = "assistant" in roles
    has_tool = "tool" in roles
    continuation = has_assistant and has_tool and all(
        role in {"system", "assistant", "tool", "developer"} for role in roles
    )
    if not continuation:
        return body, roles, False, False
    payload["messages"] = [
        *messages,
        {
            "role": "user",
            "content": SYNTHETIC_CONTINUATION,
        },
    ]
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(), roles, False, True


def _record_request(request_id: int, roles: list[str], has_user: bool, synthetic: bool) -> None:
    metrics["requests_total"] += 1
    metric = "requests_with_user_role" if has_user else "requests_missing_user_role"
    metrics[metric] += 1
    if synthetic:
        metrics["synthetic_continuation_inserted"] += 1
    if ROLE_DEBUG or synthetic:
        logger.info(
            "request=%s messages=%s roles=%s tool_tail=%s synthetic_continuation=%s",
            request_id,
            len(roles),
            ",".join(roles) if roles else "unavailable",
            bool(roles and roles[-1] == "tool"),
            synthetic,
        )


def _record_upstream(request_id: int, status_code: int, started: float) -> None:
    latency_ms = round((perf_counter() - started) * 1000, 1)
    metrics["upstream_latency_ms_total"] += latency_ms
    if status_code == 400:
        metrics["upstream_400"] += 1
    logger.debug("request=%s upstream_status=%s upstream_latency_ms=%.1f", request_id, status_code, latency_ms)


async def _close_stream(
    response: httpx.Response,
    request_id: int,
    started: float,
) -> None:
    try:
        await response.aclose()
    finally:
        _record_upstream(request_id, response.status_code, started)


@app.get("/health")
async def health() -> dict[str, object]:
    total = int(metrics["requests_total"])
    latency_total = float(metrics["upstream_latency_ms_total"])
    return {
        "status": "ok",
        "upstream": UPSTREAM_BASE_URL,
        "metrics": {
            **metrics,
            "upstream_latency_ms_average": round(latency_total / total, 1) if total else 0.0,
        },
    }


@app.get("/v1/models")
async def models(request: Request) -> Response:
    request_id = next(request_ids)
    metrics["requests_total"] += 1
    started = perf_counter()
    response = await request.app.state.upstream.get(
        f"{UPSTREAM_BASE_URL}/v1/models",
        headers=_request_headers(request),
    )
    _record_upstream(request_id, response.status_code, started)
    return Response(response.content, status_code=response.status_code, headers=_response_headers(response))


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    request_id = next(request_ids)
    original_body = await request.body()
    body, roles, has_user, synthetic = _compatibility_payload(original_body)
    _record_request(request_id, roles, has_user, synthetic)
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
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            headers=headers,
            background=BackgroundTask(_close_stream, upstream, request_id, started),
        )
    content = await upstream.aread()
    await upstream.aclose()
    _record_upstream(request_id, upstream.status_code, started)
    return Response(content, status_code=upstream.status_code, headers=headers)
