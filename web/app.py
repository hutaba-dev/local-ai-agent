"""FastAPI endpoints for the local browser agent test interface."""

from __future__ import annotations

from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from runtime.agent_runtime import BASE_URL, MODEL, AgentRuntime
from runtime.router import AGENT_CHOICES


WEB_ROOT = Path(__file__).parent
app = FastAPI(title="Local AI Agent Chat", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=WEB_ROOT / "static"), name="static")
runtime = AgentRuntime()


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12_000)
    selected_agent: str = "auto"
    session_id: str | None = None


@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    return FileResponse(WEB_ROOT / "templates" / "index.html")


@app.get("/health")
def health() -> dict[str, object]:
    try:
        response = httpx.get(f"{BASE_URL}/models", timeout=10)
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise HTTPException(status_code=503, detail="Qwen backend is unavailable") from error
    return {"status": "ok", "model": MODEL}


@app.get("/api/agents")
def agents() -> dict[str, object]:
    return {
        "agents": [
            {"id": "auto", "label": "AUTO / Main"},
            {"id": "main", "label": "Main / Secretary"},
            {"id": "coding", "label": "Coding"},
            {"id": "research", "label": "Research"},
            {"id": "server", "label": "Server"},
        ],
        "model": "Qwen3.8-27B",
    }


@app.post("/api/new-session")
def new_session() -> dict[str, str]:
    return {"session_id": runtime.new_session()}


@app.post("/api/chat")
async def chat(request: ChatRequest) -> dict[str, object]:
    if request.selected_agent not in AGENT_CHOICES:
        raise HTTPException(status_code=422, detail="unknown agent selection")
    try:
        result = await run_in_threadpool(runtime.chat, request.message, request.selected_agent, request.session_id)
    except (httpx.HTTPError, ValueError) as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    usage = result.usage or {}
    completion_tokens = usage.get("completion_tokens")
    tokens_per_second = None
    if isinstance(completion_tokens, int) and result.duration_ms > 0:
        tokens_per_second = round(completion_tokens / (result.duration_ms / 1000), 1)
    return {
        "session_id": result.session_id,
        "content": result.content,
        "activity": {
            "selected_agent": result.selected_agent,
            "routed_agent": result.route.agent,
            "direct": result.selected_agent != "auto",
            "route_summary": result.route.summary,
            "tools": result.tools,
            "duration_ms": result.duration_ms,
            "usage": usage,
            "tokens_per_second": tokens_per_second,
        },
    }