"""Small transparent router; its summary is safe to display in the UI."""

from __future__ import annotations

from dataclasses import dataclass


AGENT_CHOICES = ("auto", "main", "coding", "research", "server")


@dataclass(frozen=True)
class Route:
    agent: str
    summary: str


def route_request(message: str, selected_agent: str) -> Route:
    if selected_agent not in AGENT_CHOICES:
        raise ValueError("unknown agent selection")
    if selected_agent != "auto":
        return Route(selected_agent, f"Direct {selected_agent.title()} agent test")

    normalized = message.lower()
    if any(term in normalized for term in ("gpu", "nvidia", "vllm", "systemd", "서비스", "서버", "로그", "disk", "메모리")):
        return Route("server", "Server status or GPU diagnostics requested")
    if any(term in normalized for term in ("repository", "repo", "코드", "파일", "git", "구조", "오류", "수정")):
        return Route("coding", "Repository investigation or coding work requested")
    if any(term in normalized for term in ("문서", "자료", "분석", "연구", "정리", "qwen")):
        return Route("research", "Document research or analysis requested")
    return Route("main", "General conversation handled by Main")