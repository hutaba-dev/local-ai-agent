"""Read-only whitelist tools for the initial browser agent evaluation."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import httpx

from runtime.project_tools import ProjectTools
from runtime.academic_intelligence import academic_intelligence
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
    results = []
    if allow_local_tools:
        results = [
            _command("search_project_docs", ["find", "docs", "-type", "f", "-name", "*.md", "-print"], cwd=REPO_ROOT),
            _command("read_file", ["sed", "-n", "1,220p", "docs/model-serving.md"], cwd=REPO_ROOT),
        ]
    if search_mode != "NO_SEARCH":
        web_result = _web_search(message, search_mode)
        results.append(web_result)
        if search_mode == "DEEP_RESEARCH" and web_result.success:
            results.append(_web_sources(web_result.output))
            if _is_researcher_query(_queries(message)):
                researcher_query = (
                    researcher_identity_resolver(_queries(message), web_result.output)
                    if researcher_identity_resolver is not None
                    else _researcher_query(_queries(message), web_result.output)
                )
                results.append(_academic_intelligence(researcher_query))
            else:
                academic_result = _academic_papers(_queries(message))
                results.append(academic_result)
                results.extend(_academic_evidence_gaps(
                    _queries(message),
                    academic_result.output if academic_result.success else "[]",
                    results,
                ))
    return results


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


def _web_search(message: str | tuple[str, ...], search_mode: str) -> ToolResult:
    started = perf_counter()
    try:
        return ToolResult("web_search", True, json.dumps(search_many(_queries(message), search_mode), ensure_ascii=False), None, round((perf_counter() - started) * 1000))
    except (RuntimeError, httpx.HTTPError) as error:
        return ToolResult("web_search", False, "", str(error), round((perf_counter() - started) * 1000))


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