from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mcp import Client

from mcp_servers.academic_server import ACADEMIC_MCP
from runtime.capability_registry import capability_catalog
from mcp_servers.project_server import create_project_mcp
from runtime.mcp_host import MCPHealth, MCPHost
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

        self.assertEqual(asyncio.run(discover()), {
            "academic_resolve_researcher", "academic_search_publications",
            "academic_get_researcher_evidence", "academic_compare_source_coverage",
        })
        with patch("mcp_servers.academic_server.academic_intelligence", return_value={
            "researcher": {"canonical_name": "Ada Researcher", "identity_confidence": "HIGH"},
            "source_status": {"scopus": "AVAILABLE_FULL", "openalex": "AVAILABLE_FULL"},
            "coverage": {
                "scopus": {"status": "AVAILABLE_FULL", "reported_document_count": 10, "citation_count": 100},
                "openalex": {"status": "AVAILABLE_FULL", "reported_document_count": 8, "citation_count": 80},
            },
            "conflicts": [{"type": "publication_count_discrepancy"}],
            "representative_papers": [{"title": "Evidence", "doi": "10.1/test", "sources": ["scopus", "openalex"]}],
            "publication_candidate_count": 10,
            "merged_publication_count": 8,
        }):
            result = self.call(ACADEMIC_MCP, "academic_get_researcher_evidence", {"query": "Ada Researcher"})
        self.assertEqual(set(result.structured_content["citation_metrics_by_source"]), {"scopus", "openalex"})
        self.assertEqual(result.structured_content["representative_papers"][0]["sources"], ["scopus", "openalex"])
        self.assertNotIn("publication_candidates", result.structured_content)

    def test_academic_facade_does_not_leak_provider_errors_or_credentials(self) -> None:
        secret = "secret-provider-token"
        intelligence = {
            "researcher": {"canonical_name": "Ada Researcher", "identity_confidence": "LOW"},
            "source_status": {"scopus": "NO_ENTITLEMENT", "openalex": "AVAILABLE_FULL"},
            "source_details": {
                "scopus": {"error": f"Authorization: Bearer {secret}", "identities": []},
                "openalex": {"identities": []},
            },
            "coverage": {}, "conflicts": [],
        }
        with patch("mcp_servers.academic_server.academic_intelligence", return_value=intelligence):
            result = self.call(ACADEMIC_MCP, "academic_resolve_researcher", {"query": "Ada Researcher"})

        serialized = json.dumps(result.structured_content)
        self.assertNotIn(secret, serialized)
        self.assertNotIn("Authorization", serialized)
        self.assertEqual(result.structured_content["status"], "DEGRADED")

    def test_academic_catalog_reports_provider_states_without_requiring_paid_credentials(self) -> None:
        with patch.dict(os.environ, {"MCP_ENABLED": "true", "MCP_ACADEMIC_ENABLED": "true"}, clear=False), patch(
            "runtime.academic_intelligence.academic_source_status",
            return_value={"scopus": "UNCONFIGURED", "openalex": "AVAILABLE_FULL"},
        ):
            academic = next(item for item in capability_catalog() if item["name"] == "academic")

        self.assertTrue(academic["available"])
        self.assertEqual(academic["health"], "DEGRADED")
        self.assertEqual(academic["provider_states"]["scopus"], "UNCONFIGURED")

    def test_academic_output_is_bounded_without_returning_full_corpus(self) -> None:
        publications = [{
            "title": f"Paper {index}", "abstract": "x" * 5_000,
            "sources": ["openalex"], "source_records": [{"source": "openalex", "source_record_id": index}],
        } for index in range(30)]
        with patch("mcp_servers.academic_server.academic_papers", return_value=publications):
            result = self.call(ACADEMIC_MCP, "academic_search_publications", {"query": "topic", "limit": 10})

        serialized = json.dumps(result.structured_content, ensure_ascii=False)
        self.assertLessEqual(len(serialized), 12_000)
        self.assertLessEqual(len(result.structured_content["publications"]), 10)
        self.assertTrue(all(
            len(publication.get("abstract") or "") <= 800
            for publication in result.structured_content["publications"]
        ))

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

    def test_project_discovers_only_bounded_semantic_tools(self) -> None:
        scope = SimpleNamespace(tools=object(), owner_id="owner", project_id="prj_test", conversation_id=None)

        async def discover():
            async with Client(create_project_mcp(scope)) as client:
                return {tool.name: tool.input_schema for tool in (await client.list_tools()).tools}

        schemas = asyncio.run(discover())
        self.assertEqual(set(schemas), {
            "project_get_context", "project_search", "project_list_files", "project_read_file",
            "project_get_memories", "project_save_memory", "project_list_artifacts", "project_save_artifact",
        })
        self.assertEqual(set(schemas["project_read_file"]["properties"]), {"file_id", "offset", "max_chars"})
        self.assertNotIn("content", schemas["project_list_files"]["properties"])

    def test_project_file_reads_are_chunked_and_observations_are_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            (data_root / "projects").mkdir(parents=True)
            store = ProjectStore(root / "projects.db", data_root, require_mount=False)
            project = store.create_project("owner", "Chunked")
            saved = store.save_file("owner", project["id"], "large.txt", b"x" * 30_000, "text/plain", "x" * 30_000)
            scope = SimpleNamespace(
                tools=ProjectTools(store), owner_id="owner", project_id=project["id"], conversation_id=None,
            )
            result = self.call(create_project_mcp(scope), "project_read_file", {
                "file_id": saved["id"], "offset": 0, "max_chars": 10_000,
            })

        self.assertEqual(len(result.structured_content["content"]), 10_000)
        self.assertEqual(result.structured_content["next_offset"], 10_000)
        self.assertTrue(result.structured_content["truncated"])
        self.assertLessEqual(len(json.dumps(result.structured_content, ensure_ascii=False)), 12_000)

    def test_project_rejects_encoded_paths_and_cross_project_supersession(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            (data_root / "projects").mkdir(parents=True)
            store = ProjectStore(root / "projects.db", data_root, require_mount=False)
            first = store.create_project("owner", "First")
            second = store.create_project("owner", "Second")
            memory = store.add_memory("owner", second["id"], "fact", "Other project", "HIGH", "manual")
            scope = SimpleNamespace(
                tools=ProjectTools(store), owner_id="owner", project_id=first["id"], conversation_id=None,
            )
            server = create_project_mcp(scope)
            encoded_path = self.call(server, "project_save_artifact", {
                "name": "%2e%2e%2fstolen.md", "content": "blocked",
            })
            cross_project = self.call(server, "project_save_memory", {
                "memory_type": "fact", "content": "replacement", "supersedes_ids": [memory["id"]],
            })

        self.assertTrue(encoded_path.is_error)
        self.assertTrue(cross_project.is_error)

    def test_project_artifact_save_is_idempotent_for_same_name_and_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            (data_root / "projects").mkdir(parents=True)
            store = ProjectStore(root / "projects.db", data_root, require_mount=False)
            project = store.create_project("owner", "Artifacts")
            scope = SimpleNamespace(
                tools=ProjectTools(store), owner_id="owner", project_id=project["id"], conversation_id=None,
            )
            server = create_project_mcp(scope)
            first = self.call(server, "project_save_artifact", {"name": "report.md", "content": "result"})
            second = self.call(server, "project_save_artifact", {"name": "report.md", "content": "result"})
            listed = self.call(server, "project_list_artifacts", {})

        self.assertEqual(first.structured_content["artifact"]["id"], second.structured_content["artifact"]["id"])
        self.assertEqual(len(listed.structured_content["artifacts"]), 1)

    def test_project_storage_offline_is_not_reported_as_available(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            (data_root / "projects").mkdir(parents=True)
            store = ProjectStore(root / "projects.db", data_root, require_mount=False)
            project = store.create_project("owner", "Offline")
            scope = SimpleNamespace(
                tools=ProjectTools(store), owner_id="owner", project_id=project["id"], conversation_id=None,
            )
            shutil.rmtree(data_root)
            with patch.dict(os.environ, {"MCP_ENABLED": "true", "MCP_PROJECT_ENABLED": "true"}, clear=False):
                outcome = MCPHost().call("project_get_context", {"query": "status"}, scope)

        self.assertFalse(outcome.success)
        self.assertTrue(outcome.executed)
        self.assertEqual(outcome.status, MCPHealth.PROJECT_STORAGE_OFFLINE.value)


if __name__ == "__main__":
    unittest.main()
