from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from runtime.capability_registry import capability_catalog, detailed_tools


class CapabilityRegistryTests(unittest.TestCase):
    def test_catalog_is_compact_and_reports_unconfigured_scopes(self) -> None:
        with patch.dict(os.environ, {"MCP_ENABLED": "true"}, clear=False):
            catalog = capability_catalog(project_available=False, image_available=False)

        self.assertEqual(len(catalog), 9)
        self.assertLess(len(json.dumps(catalog)), 3_000)
        status = {item["name"]: item["status"] for item in catalog}
        self.assertEqual(status["project"], "UNCONFIGURED")
        self.assertEqual(status["media"], "DISABLED")
        self.assertEqual(status["github"], "UNCONFIGURED")
        documentation = next(item for item in catalog if item["name"] == "documentation")
        git = next(item for item in catalog if item["name"] == "git")
        self.assertEqual((documentation["provider"], documentation["permission"]), ("context7", "READ"))
        self.assertEqual((git["provider"], git["permission"]), ("local_git", "READ"))
        self.assertEqual(documentation["health"], "AVAILABLE")

    def test_only_selected_detailed_tools_are_exposed_with_ten_tool_cap(self) -> None:
        tools = detailed_tools(("time", "git", "web", "academic"))

        self.assertLessEqual(len(tools), 10)
        self.assertTrue(all(tool.capability in {"time", "git", "web", "academic"} for tool in tools))
        self.assertTrue({"get_current_time", "git_status", "search_web"} <= {tool.name for tool in tools})
        self.assertTrue(all(tool.permission == "READ" for tool in tools))

    def test_tool_cap_represents_every_selected_phase_a_capability(self) -> None:
        tools = detailed_tools(("documentation", "git", "github", "browser"))

        self.assertEqual(len(tools), 10)
        self.assertEqual({tool.capability for tool in tools}, {"documentation", "git", "github", "browser"})

    def test_phase_a2_combination_exposes_all_ten_tools(self) -> None:
        tools = detailed_tools(("github", "browser"))

        self.assertEqual(len(tools), 10)
        self.assertEqual(sum(tool.capability == "github" for tool in tools), 6)
        self.assertEqual(sum(tool.capability == "browser" for tool in tools), 4)

    def test_global_flag_disables_every_capability(self) -> None:
        with patch.dict(os.environ, {"MCP_ENABLED": "false"}, clear=False):
            catalog = capability_catalog(project_available=True, image_available=True)

        self.assertTrue(all(not item["available"] and item["status"] == "DISABLED" for item in catalog))

    def test_fetch_schema_respects_its_independent_flag(self) -> None:
        with patch.dict(os.environ, {"MCP_FETCH_ENABLED": "false"}, clear=False):
            names = {tool.name for tool in detailed_tools(("web",))}

        self.assertNotIn("fetch_page", names)

    def test_browser_requires_both_enable_and_egress_guard_flags(self) -> None:
        with patch.dict(os.environ, {"MCP_ENABLED": "true", "MCP_PLAYWRIGHT_ENABLED": "true"}, clear=False):
            without_guard = capability_catalog()
        with patch.dict(os.environ, {
            "MCP_ENABLED": "true", "MCP_PLAYWRIGHT_ENABLED": "true", "MCP_PLAYWRIGHT_EGRESS_GUARD": "true",
        }, clear=False):
            with_guard = capability_catalog()

        self.assertEqual(next(item["status"] for item in without_guard if item["name"] == "browser"), "UNCONFIGURED")
        self.assertTrue(next(item["available"] for item in with_guard if item["name"] == "browser"))


if __name__ == "__main__":
    unittest.main()
