from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mcp import Client

from mcp_servers.academic_server import ACADEMIC_MCP
from mcp_servers.project_server import create_project_mcp
from runtime.project_tools import ProjectTools
from runtime.projects import ProjectStore


class MCPPhaseBTests(unittest.TestCase):
    @staticmethod
    def call(server: object, tool: str, arguments: dict[str, object]):
        async def invoke():
            async with Client(server, raise_exceptions=False) as client:
                return await client.call_tool(tool, arguments)

        return asyncio.run(invoke())

    def test_academic_tools_are_high_level_and_preserve_provenance(self) -> None:
        async def discover():
            async with Client(ACADEMIC_MCP) as client:
                return {tool.name for tool in (await client.list_tools()).tools}

        self.assertEqual(asyncio.run(discover()), {"researcher_profile", "publication_search"})
        with patch("mcp_servers.academic_server.academic_intelligence", return_value={
            "researcher": {"canonical_name": "Ada Researcher"},
            "metrics_by_source": {"scopus": {"citations": 10}, "openalex": {"citations": 8}},
        }):
            result = self.call(ACADEMIC_MCP, "researcher_profile", {"query": "Ada Researcher"})
        self.assertEqual(set(result.structured_content["entity"]["metrics_by_source"]), {"scopus", "openalex"})

    def test_project_scope_hides_owner_and_blocks_cross_project_file_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            (data_root / "projects").mkdir(parents=True)
            store = ProjectStore(root / "projects.db", data_root, require_mount=False)
            owner_project = store.create_project("owner-a", "Allowed")
            other_project = store.create_project("owner-b", "Denied")
            owner_file = store.save_file("owner-a", owner_project["id"], "allowed.txt", b"allowed", "text/plain", "allowed")
            other_file = store.save_file("owner-b", other_project["id"], "denied.txt", b"denied", "text/plain", "denied")
            scope = SimpleNamespace(tools=ProjectTools(store), owner_id="owner-a", project_id=owner_project["id"])
            server = create_project_mcp(scope)

            allowed = self.call(server, "project_read_file", {"file_id": owner_file["id"]})
            denied = self.call(server, "project_read_file", {"file_id": other_file["id"]})

        self.assertFalse(allowed.is_error)
        self.assertEqual(allowed.structured_content["content"], "allowed")
        self.assertTrue(denied.is_error)

    def test_project_schema_never_accepts_owner_or_filesystem_paths(self) -> None:
        scope = SimpleNamespace(tools=object(), owner_id="owner", project_id="prj_test")

        async def discover():
            async with Client(create_project_mcp(scope)) as client:
                return (await client.list_tools()).tools

        for tool in asyncio.run(discover()):
            properties = tool.input_schema.get("properties", {})
            self.assertNotIn("owner_id", properties)
            self.assertNotIn("project_id", properties)
            self.assertNotIn("path", properties)


if __name__ == "__main__":
    unittest.main()
