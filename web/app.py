"""FastAPI endpoints for the local browser agent test interface."""

from __future__ import annotations

import os
import secrets
import hashlib
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from runtime.agent_runtime import BASE_URL, MODEL, AgentRuntime
from runtime.router import AGENT_CHOICES
from web.auth import SessionSigner, User, configured_user_store


WEB_ROOT = Path(__file__).parent
app = FastAPI(title="Local AI Agent Chat", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=WEB_ROOT / "static"), name="static")
runtime = AgentRuntime()
user_store = configured_user_store()
session_timeout_minutes = int(os.getenv("WEB_SESSION_IDLE_MINUTES", "15"))
session_signer = SessionSigner(os.getenv("WEB_SESSION_SECRET", secrets.token_urlsafe(32)), session_timeout_minutes)
chat_session_owners: dict[str, str] = {}
SESSION_COOKIE = "local_ai_session"


@app.middleware("http")
async def disable_ui_cache(request: Request, call_next):
    public_paths = {"/login", "/api/login", "/health"}
    if request.url.path not in public_paths and not request.url.path.startswith("/static/"):
        user = session_signer.verify(request.cookies.get(SESSION_COOKIE))
        if user is None:
            if request.url.path.startswith("/api/"):
                return JSONResponse({"detail": "login required"}, status_code=401)
            return RedirectResponse("/login", status_code=303)
        request.state.user = user
    response = await call_next(request)
    if hasattr(request.state, "user") and request.url.path != "/api/logout":
        refreshed = session_signer.renew(request.cookies.get(SESSION_COOKIE))
        if refreshed:
            response.set_cookie(
                SESSION_COOKIE,
                refreshed,
                httponly=True,
                samesite="lax",
                secure=os.getenv("WEB_SECURE_COOKIE", "0") == "1",
                max_age=session_timeout_minutes * 60,
            )
    if request.url.path in {"/", "/login"} or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store"
    return response


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12_000)
    selected_agent: str = "auto"
    session_id: str | None = None


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=40)
    password: str = Field(min_length=1, max_length=256)


def current_user(request: Request) -> User:
    return request.state.user


def chat_owner(request: Request) -> str:
    token = request.cookies.get(SESSION_COOKIE, "")
    session_key = session_signer.session_key(token)
    if session_key is None:
        raise HTTPException(status_code=401, detail="login required")
    return hashlib.sha256(session_key.encode()).hexdigest()


@app.get("/login", response_class=FileResponse)
def login_page() -> FileResponse:
    return FileResponse(WEB_ROOT / "templates" / "login.html")


@app.post("/api/login")
def login(request: LoginRequest) -> JSONResponse:
    user = user_store.authenticate(request.username, request.password)
    if user is None:
        raise HTTPException(status_code=401, detail="invalid username or password")
    response = JSONResponse({"username": user.username, "role": user.role})
    response.set_cookie(
        SESSION_COOKIE,
        session_signer.create(user),
        httponly=True,
        samesite="lax",
        secure=os.getenv("WEB_SECURE_COOKIE", "0") == "1",
        max_age=session_timeout_minutes * 60,
    )
    return response


@app.post("/api/logout")
def logout() -> JSONResponse:
    response = JSONResponse({"status": "ok"})
    response.delete_cookie(SESSION_COOKIE)
    return response


@app.get("/api/me")
def me(request: Request) -> dict[str, str]:
    user = current_user(request)
    return {"username": user.username, "role": user.role}


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
def new_session(request: Request) -> dict[str, str]:
    session_id = runtime.new_session()
    chat_session_owners[session_id] = chat_owner(request)
    return {"session_id": session_id}


@app.post("/api/chat")
async def chat(request: ChatRequest, http_request: Request) -> dict[str, object]:
    if request.selected_agent not in AGENT_CHOICES:
        raise HTTPException(status_code=422, detail="unknown agent selection")
    owner = chat_owner(http_request)
    if request.session_id and chat_session_owners.get(request.session_id) not in {None, owner}:
        raise HTTPException(status_code=403, detail="chat session belongs to another user")
    try:
        result = await run_in_threadpool(runtime.chat, request.message, request.selected_agent, request.session_id)
    except (httpx.HTTPError, ValueError) as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    chat_session_owners[result.session_id] = owner
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