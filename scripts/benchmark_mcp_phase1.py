#!/usr/bin/env python3
"""Benchmark Phase 1 MCP planner selection, execution safety, and context overhead."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime.agent_runtime import AgentRuntime, LatencyRecorder, ResearchPlan
from runtime.mcp_host import MCPCallOutcome
from runtime.tool_registry import ToolResult, execute_research_action, research_tool_catalog


@contextmanager
def flags(**values: str):
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def catalog_measurement() -> dict[str, object]:
    measurements: dict[str, dict[str, int]] = {}
    for mode, enabled in (("direct", "false"), ("mcp", "true")):
        with flags(MCP_ENABLED=enabled, MCP_SEARCH_ENABLED="true", MCP_FETCH_ENABLED="true"):
            encoded = json.dumps(research_tool_catalog(), ensure_ascii=False, separators=(",", ":"))
        measurements[mode] = {
            "tools": len(json.loads(encoded)),
            "characters": len(encoded),
            "estimated_tokens_at_4_chars": (len(encoded) + 3) // 4,
        }
    measurements["delta"] = {
        key: measurements["mcp"][key] - measurements["direct"][key]
        for key in ("tools", "characters", "estimated_tokens_at_4_chars")
    }
    return measurements


def executor_benchmark(iterations: int) -> list[dict[str, object]]:
    success = MCPCallOutcome(
        True, True, "search_web", "search-mcp", "AVAILABLE",
        {"status": "AVAILABLE", "results": [{"title": "Result", "url": "https://example.com", "snippet": "Evidence"}], "metrics": {"estimated_paid_requests": 0}},
        None, 1,
    )
    unavailable = MCPCallOutcome(False, False, "search_web", "search-mcp", "UNAVAILABLE", None, "down", 1)
    timeout = MCPCallOutcome(False, True, "search_web", "search-mcp", "DEGRADED", None, "timeout", 20_000)
    direct_result = ToolResult("web_search", True, "[]", None, 1, {"estimated_paid_requests": 0})
    cases = (
        ("A_direct_baseline", "false", success),
        ("B_mcp_success", "true", success),
        ("C_pre_execution_fallback", "true", unavailable),
        ("D_post_execution_timeout", "true", timeout),
    )
    results = []
    for name, enabled, outcome in cases:
        latencies = []
        mcp_calls = 0
        direct_calls = 0
        successful = False
        for _ in range(iterations):
            with flags(MCP_ENABLED=enabled, MCP_SEARCH_ENABLED="true", MCP_DIRECT_FALLBACK_ENABLED="true"):
                with patch("runtime.tool_registry.call_mcp_tool", return_value=outcome) as mcp_call, patch(
                    "runtime.tool_registry._direct_web_search", return_value=direct_result
                ) as direct_call:
                    started = perf_counter()
                    response = execute_research_action("SEARCH_WEB", ("benchmark topic",), "auto")
                    latencies.append((perf_counter() - started) * 1000)
            mcp_calls += mcp_call.call_count
            direct_calls += direct_call.call_count
            successful = bool(response[0]["success"])
        results.append({
            "case": name,
            "success": successful,
            "iterations": iterations,
            "mcp_attempts": mcp_calls,
            "mcp_executions": mcp_calls if enabled == "true" and outcome.executed else 0,
            "direct_calls": direct_calls,
            "duplicate_calls": max(
                0,
                (mcp_calls if enabled == "true" and outcome.executed else 0) + direct_calls - iterations,
            ),
            "paid_api_requests": 0,
            "median_executor_overhead_ms": round(statistics.median(latencies), 3),
        })
    return results


def planner_benchmark() -> list[dict[str, object]]:
    search_observation = [{
        "name": "web_search",
        "success": True,
        "output": json.dumps([{
            "title": "NVIDIA Investor News", "url": "https://investor.nvidia.com/news", "snippet": "Latest release",
        }]),
        "error": None,
        "duration_ms": 1,
    }]
    fetched_observation = [{
        "name": "web_sources",
        "success": True,
        "output": json.dumps([{
            "title": "Attention Is All You Need",
            "url": "https://arxiv.org/abs/1706.03762",
            "text": "The Transformer is based solely on attention mechanisms, dispensing with recurrence and convolutions.",
        }]),
        "error": None,
        "duration_ms": 1,
    }]
    cases = (
        ("A_current_news", "What are NVIDIA's most important announcements today?", (), {"SEARCH_WEB"}),
        ("B_researcher", "안호선 교수의 연구 역량과 대표 논문을 평가해줘", (), {"SEARCH_WEB", "SEARCH_ACADEMIC", "LOOKUP_AUTHOR"}),
        ("C_sufficient_evidence", "According to the supplied paper, what architecture does the Transformer use?", fetched_observation, {"FINAL_ANSWER"}),
        ("D_fetch_selected_source", "Verify the latest NVIDIA announcement from the primary source.", search_observation, {"FETCH_PAGE"}),
    )
    runtime = AgentRuntime()
    system_prompt = runtime._load_prompt("research")
    results = []
    try:
        with flags(MCP_ENABLED="true", MCP_SEARCH_ENABLED="true", MCP_FETCH_ENABLED="true"):
            for name, question, tools, expected in cases:
                latency = LatencyRecorder()
                decision = runtime._decide_research_action(
                    question,
                    ResearchPlan("DEEP_RESEARCH", depth="moderate", search_queries=(question,)),
                    list(tools), [], system_prompt, latency, 8, 8,
                )
                call = latency.llm_calls[-1] if latency.llm_calls else {}
                results.append({
                    "case": name,
                    "selected_action": decision.next_action,
                    "expected_actions": sorted(expected),
                    "selection_correct": decision.next_action in expected,
                    "provider": decision.provider,
                    "selected_urls": list(decision.urls),
                    "decision_summary": decision.decision_summary,
                    "input_tokens": call.get("input_tokens"),
                    "output_tokens": call.get("output_tokens"),
                    "latency_ms": call.get("total_llm_latency_ms"),
                })
    finally:
        runtime._client.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-planner", action="store_true", help="Call the configured Qwen endpoint for four planner cases.")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")

    with flags(MCP_ENABLED="true", MCP_SEARCH_ENABLED="true", MCP_FETCH_ENABLED="true"):
        premature = AgentRuntime._parse_research_decision(
            '{"next_action":"FINAL_ANSWER","unresolved_questions":["verify source"],"ready_to_answer":true}'
        )
    report = {
        "catalog_context": catalog_measurement(),
        "executor_cases": executor_benchmark(args.iterations),
        "premature_finalization_rejected": not premature.ready_to_answer,
        "planner_cases": planner_benchmark() if args.live_planner else [],
        "planner_skipped": not args.live_planner,
        "notes": [
            "Executor cases use deterministic MCP outcomes and consume no external search quota.",
            "Catalog token counts are a conservative four-characters-per-token estimate; live planner usage reports server token counts.",
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
