from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import patch

from mcp import Client

from mcp_servers.browser_server import BROWSER_MCP, _health
from mcp_servers.context7_server import CONTEXT7_MCP, _upstream_status
from mcp_servers.github_server import GITHUB_MCP, UPSTREAM_TOOLS, _parameters as github_parameters
from mcp_servers.public_web_proxy import public_addresses
from runtime.mcp_host import MCPHealth, MCPHost


class MCPExternalTests(unittest.TestCase):
    @staticmethod
    def call(server: object, tool: str, arguments: dict[str, object]):
        async def invoke():
            async with Client(server, raise_exceptions=False) as client:
                return await client.call_tool(tool, arguments)

        return asyncio.run(invoke())

    def test_facades_expose_only_approved_tools(self) -> None:
        async def discover(server: object):
            async with Client(server) as client:
                return {tool.name for tool in (await client.list_tools()).tools}

        browser = asyncio.run(discover(BROWSER_MCP))
        context7 = asyncio.run(discover(CONTEXT7_MCP))
        github = asyncio.run(discover(GITHUB_MCP))
        self.assertEqual(browser, {"browse_page", "browse_click", "browse_type", "browse_select"})
        self.assertEqual(context7, {"resolve_library_id", "query_documentation"})
        self.assertEqual(github, {
            "github_search_code", "github_get_file", "github_read_commits",
            "github_read_issues", "github_get_pull_request", "github_read_releases",
        })
        self.assertNotIn("browser_run_code_unsafe", browser)
        self.assertNotIn("browser_file_upload", browser)
        self.assertNotIn("browser_evaluate", browser)

    def test_browser_proxy_rejects_private_dns_and_non_https_ports(self) -> None:
        private_answer = [(2, 1, 6, "", ("10.20.30.40", 443))]
        with patch("mcp_servers.public_web_proxy.socket.getaddrinfo", return_value=private_answer):
            with self.assertRaises(ValueError):
                public_addresses("example.test", 443)
        with self.assertRaises(ValueError):
            public_addresses("localhost", 80)

    def test_browser_proxy_accepts_only_resolved_public_addresses(self) -> None:
        public_answer = [(2, 1, 6, "", ("93.184.216.34", 443))]
        with patch("mcp_servers.public_web_proxy.socket.getaddrinfo", return_value=public_answer):
            addresses = public_addresses("example.com", 443)

        self.assertEqual(addresses, ((2, ("93.184.216.34", 443)),))

    def test_browser_failures_use_host_health_statuses(self) -> None:
        timed_out = type("Result", (), {"is_error": True})()
        crashed = type("Result", (), {"is_error": True})()

        self.assertEqual(_health(timed_out, "Navigation timeout exceeded"), ("DEGRADED", "TIMEOUT"))
        self.assertEqual(_health(crashed, "Browser has closed unexpectedly"), ("ERROR", "BROWSER_CRASH"))

    def test_browser_blocks_private_targets_before_launch(self) -> None:
        for url in ("http://example.com", "https://127.0.0.1/private", "https://localhost/private"):
            with self.subTest(url=url):
                result = self.call(BROWSER_MCP, "browse_page", {"url": url})
                self.assertTrue(result.is_error)

    def test_browser_revalidates_redirect_and_never_starts_for_private_destination(self) -> None:
        with patch.dict(os.environ, {"MCP_PLAYWRIGHT_EGRESS_GUARD": "true"}), patch("mcp_servers.browser_server._safe_fetch", return_value=(object(), "https://127.0.0.1/private")), patch(
            "mcp_servers.browser_server.stdio_client"
        ) as upstream:
            result = self.call(BROWSER_MCP, "browse_page", {"url": "https://example.com/redirect"})

        self.assertTrue(result.is_error)
        upstream.assert_not_called()

    def test_context7_rejects_credentials_and_invalid_library_ids(self) -> None:
        secret = self.call(CONTEXT7_MCP, "resolve_library_id", {
            "library_name": "FastAPI",
            "query": "api_key=do-not-send latest lifespan",
        })
        invalid = self.call(CONTEXT7_MCP, "query_documentation", {
            "library_id": "fastapi",
            "query": "lifespan",
        })

        self.assertTrue(secret.is_error)
        self.assertTrue(invalid.is_error)

    def test_context7_normalizes_rate_limit_error_and_empty_success(self) -> None:
        self.assertEqual(_upstream_status(True, "HTTP 429 Too Many Requests"), "RATE_LIMITED")
        self.assertEqual(_upstream_status(True, "malformed upstream response"), "ERROR")
        self.assertEqual(_upstream_status(False, ""), "DEGRADED")
        self.assertEqual(_upstream_status(False, "current documentation"), "AVAILABLE")

    def test_github_remains_unconfigured_without_starting_upstream(self) -> None:
        environment = dict(os.environ)
        environment.pop("GITHUB_PERSONAL_ACCESS_TOKEN", None)
        with patch.dict(os.environ, environment, clear=True), patch(
            "mcp_servers.github_server.stdio_client"
        ) as upstream:
            result = self.call(GITHUB_MCP, "github_search_code", {"query": "MCPServer language:python"})
            outcome = MCPHost().call("github_search_code", {"query": "MCPServer language:python"})

        self.assertTrue(result.is_error)
        self.assertEqual(outcome.status, MCPHealth.UNCONFIGURED.value)
        self.assertFalse(outcome.executed)
        upstream.assert_not_called()

    def test_github_child_uses_exact_read_only_tool_allowlist(self) -> None:
        with patch.dict(os.environ, {
            "GITHUB_PERSONAL_ACCESS_TOKEN": "test-token", "GITHUB_TOOLSETS": "all",
        }, clear=False):
            parameters = github_parameters()

        self.assertIn("--read-only", parameters.args)
        self.assertIn(f"--tools={','.join(UPSTREAM_TOOLS)}", parameters.args)
        self.assertNotIn("GITHUB_TOOLSETS", parameters.env)

    def test_playwright_reads_public_javascript_page(self) -> None:
        with patch.dict(os.environ, {"MCP_PLAYWRIGHT_EGRESS_GUARD": "true"}):
            result = self.call(BROWSER_MCP, "browse_page", {
                "url": "https://example.com",
                "find_text": "Example Domain",
            })

        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content["status"], "AVAILABLE")
        self.assertIn("Example Domain", result.structured_content["relevant_text"])
        self.assertLessEqual(len(result.structured_content["relevant_text"]), 12_000)

    def test_browser_requires_egress_guard_before_process_launch(self) -> None:
        environment = dict(os.environ)
        environment.pop("MCP_PLAYWRIGHT_EGRESS_GUARD", None)
        with patch.dict(os.environ, environment, clear=True), patch("mcp_servers.browser_server.stdio_client") as upstream:
            result = self.call(BROWSER_MCP, "browse_page", {"url": "https://example.com"})

        self.assertTrue(result.is_error)
        upstream.assert_not_called()


if __name__ == "__main__":
    unittest.main()
