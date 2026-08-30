"""Small transparent router; its summary is safe to display in the UI."""

from __future__ import annotations

from dataclasses import dataclass


AGENT_CHOICES = ("auto", "main", "coding", "research", "server")


@dataclass(frozen=True)
class Route:
    agent: str
    summary: str
    search_mode: str = "NO_SEARCH"


def route_request(
    message: str,
    selected_agent: str,
    search_mode: str = "NO_SEARCH",
    recommended_agent: str = "",
) -> Route:
    if selected_agent not in AGENT_CHOICES:
        raise ValueError("unknown agent selection")
    if selected_agent != "auto":
        return Route(selected_agent, f"Direct {selected_agent.title()} agent test", search_mode)

    if search_mode != "NO_SEARCH":
        return Route("research", f"{search_mode.replace('_', ' ').title()} selected for external verification", search_mode)
    if recommended_agent in {"main", "coding", "research", "server"}:
        return Route(recommended_agent, f"KIM selected {recommended_agent.title()} expertise", search_mode)

    # Planner failure compatibility only; capability selection remains model-owned.
    normalized = message.lower()
    if any(term in normalized for term in ("gpu", "nvidia", "vllm", "systemd", "서비스", "서버", "로그", "disk", "메모리")):
        return Route("server", "Server status or GPU diagnostics requested", search_mode)
    if any(term in normalized for term in ("repository", "repo", "코드", "파일", "git", "구조", "오류", "수정")):
        return Route("coding", "Repository investigation or coding work requested", search_mode)
    if any(term in normalized for term in ("문서", "자료", "분석", "연구", "정리", "qwen")):
        return Route("research", "Document research or analysis requested", search_mode)
    return Route("main", "General conversation handled by Main", search_mode)