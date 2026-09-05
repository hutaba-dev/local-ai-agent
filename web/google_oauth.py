"""User-scoped Google OAuth authorization and token exchange helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from time import time
from urllib.parse import urlencode

import httpx


AUTHORIZATION_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
DRIVE_FILE_SCOPE = "https://www.googleapis.com/auth/drive.file"
REDIRECT_URI = "https://ahnbys.inu.ac.kr:7000/oauth/google"


class GoogleOAuthError(RuntimeError):
    pass


@dataclass(frozen=True)
class OAuthTokenResponse:
    access_token: str
    refresh_token: str | None
    expires_at: int
    scopes: tuple[str, ...]
    token_type: str


def configured() -> bool:
    enabled = os.getenv("MCP_GOOGLE_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}
    return enabled and bool(os.getenv("GOOGLE_CLIENT_ID")) and bool(os.getenv("GOOGLE_CLIENT_SECRET"))


def authorization_url(state: str) -> str:
    if not configured():
        raise GoogleOAuthError("Google OAuth is not configured")
    return f"{AUTHORIZATION_ENDPOINT}?{urlencode({
        'client_id': os.environ['GOOGLE_CLIENT_ID'],
        'redirect_uri': REDIRECT_URI,
        'response_type': 'code',
        'scope': DRIVE_FILE_SCOPE,
        'access_type': 'offline',
        'prompt': 'consent',
        'include_granted_scopes': 'true',
        'state': state,
    })}"


def _parse_token(payload: object) -> OAuthTokenResponse:
    if not isinstance(payload, dict) or not isinstance(payload.get("access_token"), str) or not payload["access_token"]:
        raise GoogleOAuthError("Google token response was invalid")
    try:
        expires_in = max(0, int(payload.get("expires_in", 3600)))
    except (TypeError, ValueError) as error:
        raise GoogleOAuthError("Google token response was invalid") from error
    scope = payload.get("scope", DRIVE_FILE_SCOPE)
    scopes = tuple(str(scope).split())
    refresh_token = payload.get("refresh_token")
    return OAuthTokenResponse(
        access_token=payload["access_token"],
        refresh_token=refresh_token if isinstance(refresh_token, str) else None,
        expires_at=int(time()) + expires_in,
        scopes=scopes,
        token_type=str(payload.get("token_type", "Bearer")),
    )


async def _token_request(data: dict[str, str]) -> OAuthTokenResponse:
    if not configured():
        raise GoogleOAuthError("Google OAuth is not configured")
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(TOKEN_ENDPOINT, data=data)
            response.raise_for_status()
            return _parse_token(response.json())
    except (httpx.HTTPError, ValueError) as error:
        raise GoogleOAuthError("Google token exchange failed") from error


async def exchange_code(code: str) -> OAuthTokenResponse:
    if not configured():
        raise GoogleOAuthError("Google OAuth is not configured")
    return await _token_request({
        "code": code,
        "client_id": os.environ["GOOGLE_CLIENT_ID"],
        "client_secret": os.environ["GOOGLE_CLIENT_SECRET"],
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
    })


async def refresh_access_token(refresh_token: str) -> OAuthTokenResponse:
    if not configured():
        raise GoogleOAuthError("Google OAuth is not configured")
    return await _token_request({
        "refresh_token": refresh_token,
        "client_id": os.environ["GOOGLE_CLIENT_ID"],
        "client_secret": os.environ["GOOGLE_CLIENT_SECRET"],
        "grant_type": "refresh_token",
    })