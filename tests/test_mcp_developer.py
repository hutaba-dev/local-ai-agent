from __future__ import annotations

import asyncio
import unittest

from mcp import Client

from mcp_servers.developer_server import DEVELOPER_MCP


class MCPDeveloperTests(unittest.TestCase):
    @staticmethod
    def call(tool: str, arguments: dict[str, object]):
        async def invoke():
            async with Client(DEVELOPER_MCP, raise_exceptions=False) as client:
                return await client.call_tool(tool, arguments)

        return asyncio.run(invoke())

    def test_time_tools_use_explicit_iana_timezones(self) -> None:
        current = self.call("get_current_time", {"timezones": ["UTC", "Asia/Seoul", "America/New_York"]})
        converted = self.call("convert_time", {
            "value": "2026-08-27T09:30:00",
            "from_timezone": "America/New_York",
            "to_timezone": "Asia/Seoul",
        })

        self.assertFalse(current.is_error)
        self.assertEqual([item["timezone"] for item in current.structured_content["times"]], [
            "UTC", "Asia/Seoul", "America/New_York",
        ])
        self.assertEqual(converted.structured_content["to"], "2026-08-27T22:30:00+09:00")

    def test_git_tools_are_read_only_and_bounded(self) -> None:
        async def discover():
            async with Client(DEVELOPER_MCP) as client:
                return await client.list_tools()

        tools = asyncio.run(discover()).tools
        names = {tool.name for tool in tools}
        self.assertEqual(names, {
            "get_current_time", "convert_time", "git_status", "git_log", "git_diff", "git_show", "git_blame",
        })
        self.assertTrue(all(name not in names for name in {"shell", "git_commit", "git_push", "git_reset", "git_checkout"}))
        result = self.call("git_log", {"limit": 2})
        self.assertFalse(result.is_error)
        self.assertLessEqual(len(result.structured_content["output"]), 12_000)

    def test_git_rejects_path_escape_and_revision_option_injection(self) -> None:
        escaped = self.call("git_diff", {"relative_path": "../../etc/passwd"})
        injected = self.call("git_show", {"revision": "--help"})

        self.assertTrue(escaped.is_error)
        self.assertTrue(injected.is_error)


if __name__ == "__main__":
    unittest.main()
