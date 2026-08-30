"""Read-only whitelist tools for the initial browser agent evaluation."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable
from urllib.parse import urlparse

import httpx

from runtime.project_tools import ProjectTools
from runtime.academic_intelligence import academic_intelligence, academic_source_status
from runtime.mcp_host import call_mcp_tool, mcp_tool_catalog, mcp_tool_enabled
from runtime.search_providers import SearchRequest, SearchRouter
from runtime.web_search import academic_papers, search_many, semantic_scholar_evidence, unpaywall_oa_locations


REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_OUTPUT_CHARS = 12_000


@dataclass(frozen=True)
class ToolResult:
    name: str
    success: bool
    output: str
    error: str | None
    duration_ms: int
    details: dict[str, object] | None = None


@dataclass(frozen=True)
class ProjectToolScope:
    tools: ProjectTools
    owner_id: str
    project_id: str


@dataclass(frozen=True)
class ResearchSourcePlan:
    intents: tuple[str, ...]
    freshness_priority: str
    academic_enabled: bool
    required_evidence: tuple[str, ...]
    selected_sources: tuple[str, ...]
    skipped_sources: tuple[str, ...]


RESEARCH_TOOL_DESCRIPTIONS = {
    "searxng": {
        "action": "SEARCH_WEB",
        "cost": "very_low",
        "freshness": "current web/news discovery; coverage varies by upstream engine",
        "description": "Low-cost default web discovery. Good first choice for broad web or news search.",
    },
    "serper": {
        "action": "SEARCH_WEB",
        "cost": "paid",
        "freshness": "current Google-style web/news results",
        "description": "Paid Google-result search. Use when stronger coverage or relevance is materially useful.",
    },
    "brave": {
        "action": "SEARCH_WEB",
        "cost": "paid",
        "freshness": "current independent web index",
        "description": "Paid independent search. Useful as a fallback or important cross-check.",
    },
    "secure_page_fetch": {
        "action": "FETCH_PAGE",
        "cost": "very_low",
        "freshness": "retrieves the current public page",
        "description": "Fetch validated public HTML pages from prior search results for primary-source detail.",
    },
    "academic_papers": {
        "action": "SEARCH_ACADEMIC",
        "cost": "low",
        "freshness": "scholarly metadata may lag very recent announcements",
        "description": "OpenAlex and related scholarly work metadata. Use when scholarly evidence itself is relevant.",
    },
    "academic_intelligence": {
        "action": "LOOKUP_AUTHOR",
        "cost": "mixed",
        "freshness": "provider-dependent scholarly identity and citation metadata",
        "description": "Scopus, Web of Science, OpenAlex, Semantic Scholar, Crossref, and ORCID evidence for researcher identity, publications, and citations.",
    },
    "scopus": {
        "action": "LOOKUP_AUTHOR", "cost": "licensed", "freshness": "curated scholarly metadata; indexing may lag",
        "description": "Scholarly author, publication, citation, and affiliation metadata within the integrated author lookup.",
    },
    "web_of_science": {
        "action": "LOOKUP_AUTHOR", "cost": "licensed", "freshness": "curated scholarly metadata; indexing may lag",
        "description": "Curated publication, researcher identity, and citation evidence within the integrated author lookup.",
    },
    "openalex": {
        "action": "SEARCH_ACADEMIC", "cost": "very_low", "freshness": "broad public scholarly graph; indexing may lag",
        "description": "Broad public works and author metadata for scholarly discovery and cross-checking.",
    },
    "semantic_scholar": {
        "action": "LOOKUP_AUTHOR", "cost": "very_low", "freshness": "public scholarly graph; provider-dependent lag",
        "description": "Independent author, paper, citation, and abstract metadata cross-check.",
    },
    "crossref": {
        "action": "LOOKUP_AUTHOR", "cost": "very_low", "freshness": "publisher-deposited DOI metadata",
        "description": "DOI and publisher metadata used to verify scholarly works and identities.",
    },
    "orcid": {
        "action": "LOOKUP_AUTHOR", "cost": "very_low", "freshness": "researcher-maintained identity records",
        "description": "Researcher identifiers and self-maintained affiliation/work records for identity resolution.",
    },
}


def research_tool_catalog() -> list[dict[str, object]]:
    router = SearchRouter()
    academic_status = academic_source_status()
    catalog: list[dict[str, object]] = []
    for name, description in RESEARCH_TOOL_DESCRIPTIONS.items():
        if name in {"searxng", "serper", "brave"} and mcp_tool_enabled("search_web"):
            continue
        if name == "secure_page_fetch" and mcp_tool_enabled("fetch_page"):
            continue
        available = True
        if name in router.providers:
            available = router.providers[name].configured()
        status = academic_status.get(name)
        if status is not None:
            available = status != "UNAVAILABLE"
        catalog.append({"name": name, "available": available, "status": status or ("AVAILABLE" if available else "UNAVAILABLE"), **description})
    if mcp_tool_enabled("search_web") or mcp_tool_enabled("fetch_page"):
        for tool in mcp_tool_catalog():
            if not mcp_tool_enabled(str(tool["name"])):
                continue
            catalog.append({
                "name": tool["name"],
                "description": tool["description"],
                "server": tool["server"],
                "cost": tool["cost"],
                "permission": tool["permission"],
                "available": tool["available"],
                "status": tool["health"],
                "action": "FETCH_PAGE" if tool["name"] == "fetch_page" else "SEARCH_WEB",
                "freshness": "current public page" if tool["name"] == "fetch_page" else "current web/news discovery",
            })
    return catalog


def execute_research_action(
    action: str,
    queries: tuple[str, ...] = (),
    provider: str = "searxng",
    prior_tools: tuple[dict[str, object], ...] = (),
    search_category: str = "web",
    freshness: str = "normal",
    urls: tuple[str, ...] = (),
) -> list[dict[str, object]]:
    if action == "SEARCH_WEB":
        if mcp_tool_enabled("search_web"):
            return [_mcp_web_search(queries, provider, search_category, freshness)]
        return [asdict(_direct_web_search(queries, provider, search_category, freshness))]
    if action == "FETCH_PAGE":
        if mcp_tool_enabled("fetch_page"):
            return [_mcp_fetch_pages(urls)]
        if urls:
            return [asdict(_web_sources(json.dumps([
                {"title": urlparse(url).hostname or "Selected source", "url": url}
                for url in urls[:3]
            ])))]
        search_tool = next(
            (tool for tool in reversed(prior_tools) if tool.get("name") == "web_search" and tool.get("success")),
            None,
        )
        if search_tool is None:
            return [asdict(ToolResult("web_sources", False, "", "no successful web search results to fetch", 0))]
        return [asdict(_web_sources(str(search_tool.get("output", ""))))]
    if action == "SEARCH_ACADEMIC":
        return [asdict(_academic_papers(queries))]
    if action == "LOOKUP_AUTHOR":
        return [asdict(_academic_intelligence("\n".join(queries)))]
    raise ValueError(f"unsupported research action: {action}")


def _mcp_call_details(outcome: object) -> dict[str, object]:
    return {
        "tool": getattr(outcome, "tool", ""),
        "server": getattr(outcome, "server", ""),
        "status": getattr(outcome, "status", "ERROR"),
        "duration_ms": getattr(outcome, "duration_ms", 0),
        "executed": getattr(outcome, "executed", False),
    }


def _direct_fallback_enabled() -> bool:
    return os.getenv("MCP_DIRECT_FALLBACK_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}


def _direct_web_search(
    queries: tuple[str, ...], provider: str, category: str, freshness: str
) -> ToolResult:
    if provider != "auto":
        return _web_search_provider(queries, provider, category, freshness)
    started = perf_counter()
    try:
        context = {
            "intents": ["CURRENT_NEWS"] if category == "news" else [],
            "freshness": "VERY_HIGH" if freshness == "high" else freshness,
        }
        results, metrics = search_many(queries, "DEEP_RESEARCH", context, include_metrics=True)
        duration_ms = round((perf_counter() - started) * 1000)
        return ToolResult(
            "web_search", True, json.dumps(results, ensure_ascii=False), None, duration_ms,
            {"execution": "conditional_fallback", "provider": "auto", **metrics},
        )
    except (RuntimeError, httpx.HTTPError) as error:
        return ToolResult(
            "web_search", False, "", str(error), round((perf_counter() - started) * 1000),
            {"execution": "conditional_fallback", "provider": "auto"},
        )


def _mcp_web_search(
    queries: tuple[str, ...], provider: str, category: str, freshness: str
) -> dict[str, object]:
    started = perf_counter()
    results: list[dict[str, object]] = []
    calls: list[dict[str, object]] = []
    seen_urls: set[str] = set()
    tool_name = "search_news" if category == "news" else "search_web"
    for query in queries[:4]:
        outcome = call_mcp_tool(tool_name, {
            "query": query,
            "max_results": 8,
            "freshness": "high" if freshness == "high" else "normal",
            "provider_hint": provider or "auto",
        })
        calls.append(_mcp_call_details(outcome))
        if not outcome.success:
            if not outcome.executed and not results and _direct_fallback_enabled():
                fallback = _direct_web_search(queries, provider or "auto", category, freshness)
                details = dict(fallback.details or {})
                details.update({"mcp_fallback": "direct_adapter", "mcp_calls": calls})
                return asdict(ToolResult(
                    fallback.name, fallback.success, fallback.output, fallback.error,
                    fallback.duration_ms, details,
                ))
            continue
        output = outcome.output or {}
        for result in output.get("results", []):
            if not isinstance(result, dict):
                continue
            url = str(result.get("url", ""))
            if url and url not in seen_urls:
                seen_urls.add(url)
                results.append(result)
    duration_ms = round((perf_counter() - started) * 1000)
    return asdict(ToolResult(
        "web_search", bool(results), json.dumps(results[:24], ensure_ascii=False),
        None if results else "MCP search returned no usable results", duration_ms,
        {"execution": "mcp", "transport": "in_process", "mcp_calls": calls},
    ))


def _mcp_fetch_pages(urls: tuple[str, ...]) -> dict[str, object]:
    started = perf_counter()
    sources: list[dict[str, object]] = []
    calls: list[dict[str, object]] = []
    for url in tuple(dict.fromkeys(urls))[:3]:
        outcome = call_mcp_tool("fetch_page", {"url": url, "extract_mode": "article"})
        calls.append(_mcp_call_details(outcome))
        if not outcome.success:
            if not outcome.executed and not sources and _direct_fallback_enabled():
                fallback = _web_sources(json.dumps([
                    {"title": urlparse(item).hostname or "Selected source", "url": item}
                    for item in urls[:3]
                ]))
                details = dict(fallback.details or {})
                details.update({"mcp_fallback": "direct_adapter", "mcp_calls": calls})
                return asdict(ToolResult(
                    fallback.name, fallback.success, fallback.output, fallback.error,
                    fallback.duration_ms, details,
                ))
            continue
        output = outcome.output or {}
        sources.append({
            "title": output.get("title", "Selected source"),
            "url": output.get("url", url),
            "text": output.get("text", ""),
            "published_date": output.get("published_at"),
            "relevance_score": 0.8,
        })
    duration_ms = round((perf_counter() - started) * 1000)
    return asdict(ToolResult(
        "web_sources", bool(sources), json.dumps(sources, ensure_ascii=False),
        None if sources else "MCP fetch returned no usable pages", duration_ms,
        {"execution": "mcp", "transport": "in_process", "mcp_calls": calls},
    ))


def research_source_plan(query: str | tuple[str, ...]) -> ResearchSourcePlan:
    """Return non-semantic metadata only when legacy callers lack an LLM plan."""
    return ResearchSourcePlan(
        ("FALLBACK",), "NORMAL", False, (), ("General Web Search",),
        ("Specialized sources require an LLM decision",),
    )


def run_agent_tools(
    agent: str,
    message: str | tuple[str, ...],
    search_mode: str = "NO_SEARCH",
    allow_local_tools: bool = True,
    project_scope: ProjectToolScope | None = None,
    researcher_identity_resolver: Callable[[tuple[str, ...], str], str] | None = None,
) -> list[dict[str, object]]:
    results: list[ToolResult] = []
    if agent == "research":
        results.extend(_research_tools(message, search_mode, allow_local_tools, researcher_identity_resolver))
    elif allow_local_tools:
        tools = {
            "coding": _coding_tools,
            "server": _server_tools,
        }.get(agent, lambda _message, _search_mode: [])
        results.extend(tools(message, search_mode))
    if project_scope is not None:
        results.append(_project_search(message, project_scope))
    return [asdict(result) for result in results]


def _project_search(message: str | tuple[str, ...], scope: ProjectToolScope) -> ToolResult:
    started = perf_counter()
    query = message[0] if isinstance(message, tuple) else message
    try:
        output = scope.tools.hybrid_search(scope.owner_id, scope.project_id, query)
        return ToolResult(
            "project_hybrid_search",
            True,
            json.dumps(output, ensure_ascii=False),
            None,
            round((perf_counter() - started) * 1000),
            {"semantic_available": bool(output["semantic_available"])},
        )
    except (RuntimeError, ValueError) as error:
        return ToolResult(
            "project_hybrid_search",
            False,
            "",
            str(error),
            round((perf_counter() - started) * 1000),
        )


def _coding_tools(message: str, search_mode: str) -> list[ToolResult]:
    return [
        _command("list_files", ["find", ".", "-maxdepth", "2", "-type", "f", "-not", "-path", "./.git/*"], cwd=REPO_ROOT),
        _command("search_files", ["git", "grep", "-n", "Qwen3.8-27B", "--", "README.md", "docs"], cwd=REPO_ROOT),
        _command("read_file", ["sed", "-n", "1,220p", "README.md"], cwd=REPO_ROOT),
        _command("git_status", ["git", "status", "--short", "--branch"], cwd=REPO_ROOT),
        _command("git_diff", ["git", "diff", "--stat"], cwd=REPO_ROOT),
    ]


def _research_tools(
    message: str,
    search_mode: str,
    allow_local_tools: bool = True,
    researcher_identity_resolver: Callable[[tuple[str, ...], str], str] | None = None,
) -> list[ToolResult]:
    plan = research_source_plan(message)
    results: list[ToolResult] = []
    if allow_local_tools:
        results.extend([
            _command("search_project_docs", ["find", "docs", "-type", "f", "-name", "*.md", "-print"], cwd=REPO_ROOT),
            _command("read_file", ["sed", "-n", "1,220p", "docs/model-serving.md"], cwd=REPO_ROOT),
        ])
    if search_mode != "NO_SEARCH":
        results.append(ToolResult(
            "research_source_plan", True, json.dumps(asdict(plan), ensure_ascii=False), None, 0,
            {"cost_gate": "Only sources with a realistic chance of answering the query are enabled."},
        ))
        routed_queries = _source_queries(message, plan)
        web_result = _web_search(routed_queries, search_mode, plan)
        results.append(web_result)
        if search_mode == "DEEP_RESEARCH" and web_result.success:
            results.append(_web_sources(web_result.output))
            if plan.academic_enabled and _is_researcher_query(_queries(message)):
                researcher_query = (
                    researcher_identity_resolver(_queries(message), web_result.output)
                    if researcher_identity_resolver is not None
                    else _researcher_query(_queries(message), web_result.output)
                )
                results.append(_academic_intelligence(researcher_query))
            elif plan.academic_enabled:
                academic_result = _academic_papers(_queries(message))
                results.append(academic_result)
                results.extend(_academic_evidence_gaps(
                    _queries(message),
                    academic_result.output if academic_result.success else "[]",
                    results,
                ))
    return results


def _source_queries(message: str | tuple[str, ...], plan: ResearchSourcePlan) -> tuple[str, ...]:
    requested_queries = list(_queries(message))
    queries = requested_queries[1:] if len(requested_queries) > 1 else requested_queries
    return tuple(dict.fromkeys(query[:500] for query in queries))


def _server_tools(message: str, search_mode: str) -> list[ToolResult]:
    return [
        _command("nvidia_smi", ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"]),
        _command("systemctl_status_qwen_vllm", ["systemctl", "is-active", "qwen-vllm.service"]),
        _command("df", ["df", "-h", "/"]),
        _command("free", ["free", "-h"]),
    ]


def _command(name: str, command: list[str], cwd: Path | None = None) -> ToolResult:
    started = perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        output = completed.stdout[-MAX_OUTPUT_CHARS:].strip()
        error = completed.stderr[-MAX_OUTPUT_CHARS:].strip() or None
        return ToolResult(name, completed.returncode == 0, output, error, round((perf_counter() - started) * 1000))
    except (OSError, subprocess.TimeoutExpired) as error:
        return ToolResult(name, False, "", str(error), round((perf_counter() - started) * 1000))


def _queries(message: str | tuple[str, ...]) -> tuple[str, ...]:
    return (message,) if isinstance(message, str) else message


def _web_search(
    message: str | tuple[str, ...],
    search_mode: str,
    plan: ResearchSourcePlan | None = None,
) -> ToolResult:
    started = perf_counter()
    try:
        context = {
            "intents": list(plan.intents) if plan is not None else [],
            "freshness": plan.freshness_priority if plan is not None else None,
            "required_evidence": list(plan.required_evidence) if plan is not None else [],
            "requires_primary": bool(plan and any(
                intent in plan.intents for intent in ("MARKET_FINANCE", "COMPANY_RESEARCH", "TECHNICAL_RESEARCH")
            )),
        }
        results, search_metrics = search_many(_queries(message), search_mode, context, include_metrics=True)
        if plan is not None:
            results = _rank_relevant_web_results(results, plan)
        duration_ms = round((perf_counter() - started) * 1000)
        return ToolResult(
            "web_search", True, json.dumps(results, ensure_ascii=False), None, duration_ms,
            {"execution": "conditional_fallback", "wall_time_ms": duration_ms, **search_metrics},
        )
    except (RuntimeError, httpx.HTTPError) as error:
        return ToolResult("web_search", False, "", str(error), round((perf_counter() - started) * 1000))


def _web_search_provider(
    queries: tuple[str, ...], provider_name: str, category: str = "web", freshness: str = "normal"
) -> ToolResult:
    started = perf_counter()
    router = SearchRouter()
    provider = router.providers.get(provider_name)
    if provider is None:
        return ToolResult("web_search", False, "", f"unknown search provider: {provider_name}", 0)
    if not provider.configured():
        return ToolResult("web_search", False, "", f"search provider unavailable: {provider_name}", 0)
    results: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    seen_urls: set[str] = set()
    try:
        for query in queries[:4]:
            response = provider.search(SearchRequest(
                query=query,
                category="news" if category == "news" else "web",
                freshness="VERY_HIGH" if freshness == "high" else None,
                count=8,
            ))
            diagnostics.append({
                "provider": provider_name,
                "status": response.status.value,
                "latency_ms": response.latency_ms,
                "error": response.error,
            })
            if response.status.value != "AVAILABLE":
                continue
            for item in response.results:
                value = item.to_dict()
                url = str(value.get("url", ""))
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    results.append(value)
        if not results:
            raise RuntimeError(f"{provider_name} returned no usable search results")
        duration_ms = round((perf_counter() - started) * 1000)
        return ToolResult(
            "web_search", True, json.dumps(results[:24], ensure_ascii=False), None, duration_ms,
            {"provider": provider_name, "query_count": len(queries[:4]), "requests": diagnostics},
        )
    except (RuntimeError, httpx.HTTPError) as error:
        return ToolResult(
            "web_search", False, "", str(error), round((perf_counter() - started) * 1000),
            {"provider": provider_name, "requests": diagnostics, "failure_reason": _failure_reason(error)},
        )


def _rank_relevant_web_results(
    results: list[dict[str, str]], plan: ResearchSourcePlan
) -> list[dict[str, object]]:
    ranked: list[dict[str, object]] = []
    for result in results:
        provider_score = result.get("score")
        score = float(provider_score) if isinstance(provider_score, (int, float)) else 0.5
        ranked.append({**result, "relevance_score": score})
    ranked.sort(key=lambda result: float(result["relevance_score"]), reverse=True)
    diversified: list[dict[str, object]] = []
    host_counts: dict[str, int] = {}
    for result in ranked:
        host = urlparse(str(result.get("url", ""))).netloc.lower().removeprefix("www.")
        if host_counts.get(host, 0) >= 2:
            continue
        host_counts[host] = host_counts.get(host, 0) + 1
        diversified.append(result)
        if len(diversified) == 24:
            break
    return diversified


def _web_sources(search_output: str) -> ToolResult:
    started = perf_counter()
    try:
        results = json.loads(search_output)
        if not isinstance(results, list):
            raise ValueError("web search returned an invalid result list")
        from runtime.web_search import fetch_sources

        sources, fetches = fetch_sources(results, include_metrics=True)
        if not sources:
            raise RuntimeError("no public HTML sources could be fetched")
        duration_ms = round((perf_counter() - started) * 1000)
        return ToolResult(
            "web_sources",
            True,
            json.dumps(sources, ensure_ascii=False),
            None,
            duration_ms,
            {"execution": "sequential", "wall_time_ms": duration_ms, "fetches": fetches},
        )
    except (RuntimeError, ValueError, json.JSONDecodeError, httpx.HTTPError) as error:
        duration_ms = round((perf_counter() - started) * 1000)
        return ToolResult(
            "web_sources",
            False,
            "",
            str(error),
            duration_ms,
            {"execution": "sequential", "wall_time_ms": duration_ms, "failure_reason": _failure_reason(error)},
        )


def _academic_papers(queries: tuple[str, ...]) -> ToolResult:
    started = perf_counter()
    try:
        diagnostics: list[dict[str, object]] = []
        papers = academic_papers(queries, diagnostics=diagnostics)
        if not papers:
            raise RuntimeError("OpenAlex returned no matching academic works")
        duration_ms = round((perf_counter() - started) * 1000)
        return ToolResult(
            "academic_papers",
            True,
            json.dumps(papers, ensure_ascii=False),
            None,
            duration_ms,
            {"execution": "sequential", "wall_time_ms": duration_ms, "query_count": len(queries), "requests": diagnostics},
        )
    except (RuntimeError, httpx.HTTPError) as error:
        duration_ms = round((perf_counter() - started) * 1000)
        return ToolResult(
            "academic_papers",
            False,
            "",
            str(error),
            duration_ms,
            {"execution": "sequential", "wall_time_ms": duration_ms, "failure_reason": _failure_reason(error), "requests": diagnostics if "diagnostics" in locals() else []},
        )


def _academic_intelligence(query: str) -> ToolResult:
    started = perf_counter()
    try:
        intelligence = academic_intelligence(query)
        duration_ms = round((perf_counter() - started) * 1000)
        coverage = intelligence.get("coverage")
        details = {
            "execution": "parallel",
            "wall_time_ms": duration_ms,
            "source_status": intelligence.get("source_status", {}),
            "providers_called": intelligence.get("selection_policy", {}).get("providers_called", []),
            "identity_sources": intelligence.get("researcher", {}).get("identity_sources", []),
            "identity_confidence": intelligence.get("researcher", {}).get("identity_confidence", "UNKNOWN"),
            "publication_candidates": {
                source: values.get("reported_document_count") or values.get("publication_count", 0)
                for source, values in coverage.items()
                if isinstance(values, dict)
            } if isinstance(coverage, dict) else {},
            "coverage_conflicts": len(intelligence.get("conflicts", [])),
            "merged_verified_corpus": intelligence.get("merged_publication_count", 0),
            "representative_papers": len(intelligence.get("representative_papers", [])),
            "cache_hit": intelligence.get("cache_hit", False),
        }
        return ToolResult(
            "academic_intelligence", True, json.dumps(intelligence, ensure_ascii=False), None, duration_ms, details
        )
    except (RuntimeError, ValueError, httpx.HTTPError) as error:
        duration_ms = round((perf_counter() - started) * 1000)
        return ToolResult(
            "academic_intelligence", False, "", str(error), duration_ms,
            {"execution": "parallel", "wall_time_ms": duration_ms, "failure_reason": _failure_reason(error)},
        )


def _is_researcher_query(queries: tuple[str, ...]) -> bool:
    text = " ".join(queries)
    return bool(re.search(
        r"(?:교수|박사|연구자|학자|\bprofessor\b|\bresearcher\b|\bacademic\b|\bscientist\b|"
        r"h[- ]?index|citation count|피인용|연구 역량|학술 실적)",
        text,
        re.IGNORECASE,
    ))


def _academic_evidence_gaps(
    queries: tuple[str, ...],
    academic_output: str,
    existing_results: list[ToolResult],
) -> list[ToolResult]:
    try:
        papers = json.loads(academic_output)
        if not isinstance(papers, list):
            return []
    except json.JSONDecodeError:
        return []
    source_count = next((len(json.loads(result.output)) for result in existing_results if result.name == "web_sources" and result.success), 0)
    evidence: list[ToolResult] = []
    if _needs_s2_cross_check(papers, queries):
        evidence.append(_semantic_scholar(_researcher_query(queries), _search_title_hints(existing_results)))
    if source_count < 2 and any(isinstance(paper, dict) and isinstance(paper.get("doi"), str) for paper in papers):
        evidence.append(_unpaywall_locations(papers))
    return evidence


def _needs_s2_cross_check(papers: list[object], queries: tuple[str, ...]) -> bool:
    citation_request = any(
        term in " ".join(queries).lower()
        for term in ("citation", "citations", "h-index", "h index", "인용", "피인용")
    )
    return citation_request or len(papers) < 3 or sum(
        1 for paper in papers if isinstance(paper, dict) and isinstance(paper.get("cited_by_count"), int)
    ) < 3


def _researcher_query(queries: tuple[str, ...], web_output: str = "") -> str:
    original = queries[0][:500]
    if not re.search(r"[가-힣]", original):
        return original
    alias = next(
        (candidate for query in queries[1:] if (candidate := _latin_person_name(query))),
        None,
    )
    if alias is None:
        try:
            web_results = json.loads(web_output)
        except json.JSONDecodeError:
            web_results = []
        if isinstance(web_results, list):
            for item in web_results:
                if not isinstance(item, dict):
                    continue
                text = f"{item.get('title', '')} {item.get('description', '')}"
                alias = _latin_person_name(text)
                if alias:
                    break
    return f"{original}\nAcademic alias: {alias}" if alias else original


def _latin_person_name(text: str) -> str | None:
    boundary_terms = {
        "college", "department", "engineering", "incheon", "institute", "korea", "korean",
        "laboratory", "mechanical", "national", "professor", "research", "school", "seoul", "university",
    }
    tokens = re.findall(r"\b[A-Z][A-Za-z'-]*\b", text)
    candidate: list[str] = []
    for token in tokens:
        if token.isupper() and len(token) > 1:
            candidate = []
            continue
        if token.casefold() in boundary_terms:
            if len(candidate) >= 2:
                break
            candidate = []
            continue
        candidate.append(token)
        if len(candidate) == 3:
            break
    return " ".join(candidate) if 2 <= len(candidate) <= 3 else None


def _search_title_hints(results: list[ToolResult]) -> tuple[str, ...]:
    search_result = next((result for result in results if result.name == "web_search" and result.success), None)
    if search_result is None:
        return ()
    try:
        items = json.loads(search_result.output)
    except json.JSONDecodeError:
        return ()
    if not isinstance(items, list):
        return ()
    return tuple(item["title"] for item in items[:5] if isinstance(item, dict) and isinstance(item.get("title"), str))


def _semantic_scholar(query: str, author_hints: tuple[str, ...]) -> ToolResult:
    started = perf_counter()
    diagnostics: list[dict[str, object]] = []
    evidence = semantic_scholar_evidence(query, author_hints, diagnostics)
    duration_ms = round((perf_counter() - started) * 1000)
    if evidence is None:
        return ToolResult(
            "semantic_scholar",
            False,
            "",
            "Semantic Scholar unavailable or no suitable author candidate",
            duration_ms,
            {"wall_time_ms": duration_ms, "failure_reason": "unavailable_or_no_matching_author", "requests": diagnostics},
        )
    return ToolResult("semantic_scholar", True, json.dumps(evidence, ensure_ascii=False), None, duration_ms, {"wall_time_ms": duration_ms, "requests": diagnostics})


def _failure_reason(error: BaseException) -> str:
    if isinstance(error, httpx.TimeoutException):
        return "timeout"
    if isinstance(error, httpx.HTTPStatusError):
        return f"http_{error.response.status_code}"
    if isinstance(error, httpx.HTTPError):
        return "network_error"
    if isinstance(error, json.JSONDecodeError):
        return "parsing_error"
    return "no_results_or_invalid_response"


def _unpaywall_locations(papers: list[object]) -> ToolResult:
    started = perf_counter()
    typed_papers = [paper for paper in papers if isinstance(paper, dict)]
    locations = unpaywall_oa_locations(typed_papers)
    if not locations:
        return ToolResult("unpaywall_oa_location", False, "", "Unpaywall unavailable, not configured, or no legal OA location found", round((perf_counter() - started) * 1000))
    return ToolResult("unpaywall_oa_location", True, json.dumps(locations, ensure_ascii=False), None, round((perf_counter() - started) * 1000))