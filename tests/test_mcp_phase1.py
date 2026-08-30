from __future__ import annotations

import asyncio
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mcp import Client

from mcp_servers.fetch_server import fetch_mcp
from mcp_servers.search_server import search_mcp
from runtime.agent_runtime import AgentRuntime
from runtime.capability_registry import TOOL_SPECS
from runtime.mcp_host import MCPCallOutcome, MCPHealth, MCPHost
from runtime.search_providers import SearchBatch
from runtime.tool_registry import ToolResult, execute_research_action, research_tool_catalog


class MCPPhase1Tests(unittest.TestCase):
    @staticmethod
    def call(server: object, tool: str, arguments: dict[str, object]):
        async def invoke():
            async with Client(server, raise_exceptions=False) as client:
                return await client.call_tool(tool, arguments)

        return asyncio.run(invoke())

    def test_discovery_exposes_compact_high_level_tools(self) -> None:
        host = MCPHost()
        catalog = host.catalog()

        self.assertEqual({tool["name"] for tool in catalog}, set(TOOL_SPECS))
        self.assertTrue(all(tool["input_schema"] for tool in catalog))
        self.assertTrue(any("discovery metadata" in str(tool["description"]) for tool in catalog))
        self.assertTrue(any("public webpage" in str(tool["description"]) for tool in catalog))

    def test_search_web_calls_existing_router_once_and_returns_normalized_result(self) -> None:
        batch = SearchBatch(({
            "title": "Result", "url": "https://example.com/result", "snippet": "Current evidence",
            "published_at": "2026-08-27", "source": "Example", "provider": "searxng",
            "category": "web", "rank": 1, "providers_seen": ["searxng"],
        },), {"estimated_paid_requests": 0, "providers": {"searxng": {"status": "AVAILABLE"}}})
        with patch("mcp_servers.search_server.SearchRouter") as router_type:
            router_type.return_value.search.return_value = batch
            result = self.call(search_mcp, "search_web", {"query": "current topic", "max_results": 5})

        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content["status"], "AVAILABLE")
        self.assertEqual(result.structured_content["results"][0]["provider"], "searxng")
        router_type.return_value.search.assert_called_once()

    def test_fetch_page_reuses_bounded_secure_extractor(self) -> None:
        with patch("mcp_servers.fetch_server.fetch_sources", return_value=([{
            "title": "Evidence", "url": "https://example.com/final", "text": "evidence" * 40,
        }], [{"success": True, "text_length": 320}])) as fetch:
            result = self.call(fetch_mcp, "fetch_page", {"url": "https://example.com/start"})

        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content["status"], "AVAILABLE")
        self.assertEqual(result.structured_content["url"], "https://example.com/final")
        fetch.assert_called_once_with(
            [{"title": "example.com", "url": "https://example.com/start"}], limit=1, include_metrics=True,
        )

    def test_fetch_page_blocks_malformed_local_private_and_oversized_arguments(self) -> None:
        for url in ("not-a-url", "https://localhost/private", "https://127.0.0.1/private"):
            with self.subTest(url=url):
                result = self.call(fetch_mcp, "fetch_page", {"url": url})
                self.assertFalse(result.is_error)
                self.assertEqual(result.structured_content["status"], "ERROR")
                self.assertEqual(result.structured_content["text"], "")
        oversized = self.call(fetch_mcp, "fetch_page", {"url": "https://example.com/" + "x" * 2_000})
        self.assertTrue(oversized.is_error)

    def test_host_handles_server_down_unknown_tool_and_invalid_arguments(self) -> None:
        with patch.dict(os.environ, {"MCP_ENABLED": "true", "MCP_SEARCH_ENABLED": "true"}, clear=False):
            host = MCPHost()
            host._servers.pop("search-mcp")
            down = host.call("search_web", {"query": "topic"})
            unknown = host.call("missing_tool", {})
            invalid = MCPHost().call("search_web", {})

        self.assertFalse(down.success)
        self.assertFalse(down.executed)
        self.assertEqual(down.status, MCPHealth.UNAVAILABLE.value)
        self.assertFalse(unknown.success)
        self.assertFalse(unknown.executed)
        self.assertFalse(invalid.success)
        self.assertTrue(invalid.executed)
        self.assertEqual(invalid.status, MCPHealth.ERROR.value)
        self.assertNotIn("Traceback", invalid.error or "")

    def test_host_does_not_retry_ambiguous_result_timeout_or_interruption(self) -> None:
        class FakeClient:
            behavior: object

            def __init__(self, *_args, **_kwargs) -> None:
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args) -> None:
                return None

            async def call_tool(self, *_args, **_kwargs):
                if isinstance(self.behavior, BaseException):
                    raise self.behavior
                return self.behavior

        cases = (
            (SimpleNamespace(is_error=False, structured_content=["invalid"]), MCPHealth.ERROR.value),
            (asyncio.TimeoutError(), MCPHealth.DEGRADED.value),
            (ConnectionError("interrupted"), MCPHealth.ERROR.value),
        )
        for behavior, expected_status in cases:
            with self.subTest(status=expected_status, behavior=type(behavior).__name__):
                FakeClient.behavior = behavior
                with patch.dict(os.environ, {"MCP_ENABLED": "true", "MCP_SEARCH_ENABLED": "true"}, clear=False), patch("runtime.mcp_host.Client", FakeClient):
                    outcome = MCPHost().call("search_web", {"query": "topic"})

                self.assertFalse(outcome.success)
                self.assertTrue(outcome.executed)
                self.assertEqual(outcome.status, expected_status)

    def test_executor_uses_mcp_without_duplicate_direct_search(self) -> None:
        output = {
            "status": "AVAILABLE",
            "results": [{"title": "Result", "url": "https://example.com", "snippet": "Evidence"}],
            "metrics": {"estimated_paid_requests": 0},
        }
        outcome = MCPCallOutcome(True, True, "search_web", "search-mcp", "AVAILABLE", output, None, 4)
        with patch.dict(os.environ, {"MCP_ENABLED": "true", "MCP_SEARCH_ENABLED": "true"}, clear=False):
            with patch("runtime.tool_registry.call_mcp_tool", return_value=outcome) as mcp_call, patch(
                "runtime.tool_registry._direct_web_search"
            ) as direct:
                result = execute_research_action("SEARCH_WEB", ("current topic",), "auto")

        self.assertTrue(result[0]["success"])
        self.assertEqual(json.loads(result[0]["output"])[0]["title"], "Result")
        mcp_call.assert_called_once()
        direct.assert_not_called()

    def test_executor_falls_back_once_only_before_mcp_execution(self) -> None:
        direct_result = ToolResult("web_search", True, "[]", None, 2)
        unavailable = MCPCallOutcome(
            False, False, "search_web", "search-mcp", "UNAVAILABLE", None, "down", 1,
        )
        uncertain = MCPCallOutcome(
            False, True, "search_web", "search-mcp", "DEGRADED", None, "timeout", 21_000,
        )
        flags = {"MCP_ENABLED": "true", "MCP_SEARCH_ENABLED": "true", "MCP_DIRECT_FALLBACK_ENABLED": "true"}
        with patch.dict(os.environ, flags, clear=False), patch(
            "runtime.tool_registry._direct_web_search", return_value=direct_result
        ) as direct, patch("runtime.tool_registry.call_mcp_tool", return_value=unavailable):
            fallback = execute_research_action("SEARCH_WEB", ("topic",), "auto")
        self.assertTrue(fallback[0]["success"])
        direct.assert_called_once()

        with patch.dict(os.environ, flags, clear=False), patch(
            "runtime.tool_registry._direct_web_search", return_value=direct_result
        ) as direct, patch("runtime.tool_registry.call_mcp_tool", return_value=uncertain):
            failed = execute_research_action("SEARCH_WEB", ("topic",), "auto")
        self.assertFalse(failed[0]["success"])
        direct.assert_not_called()

    def test_feature_flag_catalog_replaces_provider_tools_with_mcp_capabilities(self) -> None:
        with patch.dict(os.environ, {"MCP_ENABLED": "true", "MCP_SEARCH_ENABLED": "true", "MCP_FETCH_ENABLED": "true"}, clear=False):
            catalog = research_tool_catalog()

        names = {str(tool["name"]) for tool in catalog}
        self.assertTrue({"search_web", "search_news", "fetch_page"} <= names)
        self.assertTrue({"searxng", "serper", "brave", "secure_page_fetch"}.isdisjoint(names))
        self.assertTrue({"git_status", "query_documentation", "resolve_library_id"}.isdisjoint(names))

    def test_disabled_mcp_does_not_start_discovery(self) -> None:
        with patch.dict(os.environ, {"MCP_ENABLED": "false"}, clear=False), patch(
            "runtime.tool_registry.mcp_tool_catalog"
        ) as catalog_call:
            names = {str(tool["name"]) for tool in research_tool_catalog()}

        catalog_call.assert_not_called()
        self.assertTrue({"searxng", "serper", "brave", "secure_page_fetch"} <= names)

    def test_host_enforces_feature_flags_at_invocation_time(self) -> None:
        with patch.dict(os.environ, {"MCP_ENABLED": "false"}, clear=False):
            global_disabled = MCPHost().call("get_current_time", {})
        with patch.dict(os.environ, {"MCP_ENABLED": "true", "MCP_TIME_ENABLED": "false"}, clear=False):
            capability_disabled = MCPHost().call("get_current_time", {})

        self.assertFalse(global_disabled.executed)
        self.assertEqual(global_disabled.status, MCPHealth.UNCONFIGURED.value)
        self.assertFalse(capability_disabled.executed)

    def test_qwen_decision_supports_auto_provider_and_explicit_fetch_urls(self) -> None:
        search = AgentRuntime._parse_research_decision(
            '{"next_action":"SEARCH_WEB","queries":["topic"],"provider":"auto","ready_to_answer":false}'
        )
        fetch = AgentRuntime._parse_research_decision(
            '{"next_action":"FETCH_PAGE","queries":[],"urls":["https://example.com/report"],'
            '"provider":"","ready_to_answer":false}'
        )

        self.assertEqual(search.provider, "auto")
        self.assertEqual(fetch.urls, ("https://example.com/report",))


if __name__ == "__main__":
    unittest.main()
