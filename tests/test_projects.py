from __future__ import annotations

import tempfile
import unittest
import shutil
import os
from pathlib import Path
from unittest.mock import patch

from runtime.projects import (
    ProjectConversationImportError,
    ProjectNotFoundError,
    ProjectPathError,
    ProjectStorageOfflineError,
    ProjectStore,
)
from runtime.project_tools import ProjectTools
from runtime.tool_registry import ProjectToolScope, run_agent_tools


class ProjectStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.data_root = root / "data"
        (self.data_root / "projects").mkdir(parents=True)
        self.database = root / "state" / "projects.db"
        self.store = ProjectStore(self.database, self.data_root, require_mount=False)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_project_create_list_and_authorization(self) -> None:
        project = self.store.create_project("owner-a", "Persistent Research", "Long-running work")
        self.assertEqual(self.store.list_projects("owner-a")[0]["id"], project["id"])
        self.assertTrue((self.data_root / "projects" / project["id"] / "files").is_dir())
        with self.assertRaises(ProjectNotFoundError):
            self.store.get_project("owner-b", project["id"])

    def test_project_create_initializes_empty_mounted_layout(self) -> None:
        empty_root = Path(self.temporary.name) / "empty-data"
        empty_root.mkdir()
        store = ProjectStore(self.database, empty_root, require_mount=False)

        project = store.create_project("owner", "First project")

        self.assertTrue((empty_root / "projects" / project["id"] / "files").is_dir())

    def test_project_delete_removes_owned_database_and_storage_records(self) -> None:
        project = self.store.create_project("owner", "Disposable")
        conversation = self.store.create_conversation("owner", project["id"], "Delete me")
        message = self.store.add_message("owner", project["id"], conversation["id"], "user", "indexed message")
        memory = self.store.add_memory("owner", project["id"], "fact", "indexed memory")
        self.store.save_file(
            "owner", project["id"], "notes.txt", b"indexed file", "text/plain", "indexed file",
            conversation["id"],
        )
        project_root = self.data_root / "projects" / project["id"]

        with self.assertRaises(ProjectNotFoundError):
            self.store.delete_project("other-owner", project["id"])
        self.assertTrue(project_root.exists())

        self.store.delete_project("owner", project["id"])

        self.assertFalse(project_root.exists())
        self.assertEqual(self.store.list_projects("owner"), [])
        with self.assertRaises(ProjectNotFoundError):
            self.store.get_project("owner", project["id"])
        with self.store._connect() as connection:
            for table in ("conversations", "files", "memories", "artifacts", "project_events"):
                count = connection.execute(f"SELECT COUNT(*) FROM {table} WHERE project_id = ?", (project["id"],)).fetchone()[0]
                self.assertEqual(count, 0, table)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM messages WHERE id = ?", (message,)).fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM memory_sources WHERE memory_id = ?", (memory["id"],)).fetchone()[0], 0)
            for table in ("message_fts", "memory_fts", "file_chunk_fts"):
                count = connection.execute(f"SELECT COUNT(*) FROM {table} WHERE project_id = ?", (project["id"],)).fetchone()[0]
                self.assertEqual(count, 0, table)

    def test_conversation_messages_survive_store_restart(self) -> None:
        project = self.store.create_project("owner", "Persistence")
        conversation = self.store.create_conversation("owner", project["id"], "Experiment")
        self.store.add_message("owner", project["id"], conversation["id"], "user", "Pressure is 12 bar")
        restarted = ProjectStore(self.database, self.data_root, require_mount=False)
        messages = restarted.list_messages("owner", project["id"], conversation["id"])
        self.assertEqual(messages[0]["content"], "Pressure is 12 bar")

    def test_create_project_imports_conversation_with_provenance_and_research_metadata(self) -> None:
        source = [
            {"role": "user", "content": "NVIDIA를 조사해줘"},
            {
                "role": "assistant",
                "content": "조사 결과",
                "metadata": {"research_result": {"body_markdown": "# 조사 결과", "sources": []}},
            },
        ]

        project, conversation = self.store.create_project_with_imported_conversation(
            "owner", "안호선", "source-session", source
        )

        messages = self.store.list_messages("owner", project["id"], conversation["id"])
        events = self.store.list_events("owner", project["id"])
        self.assertEqual([message["content"] for message in messages], ["NVIDIA를 조사해줘", "조사 결과"])
        self.assertEqual(messages[1]["tool_metadata"][0]["type"], "research_result")
        imported = next(event for event in events if event["event_type"] == "conversation_imported")
        self.assertEqual(imported["details"], {
            "source": "general_chat", "source_conversation_id": "source-session",
        })

    def test_import_failure_rolls_back_created_project(self) -> None:
        with patch.object(self.store, "add_message", side_effect=RuntimeError("write failed")):
            with self.assertRaises(ProjectConversationImportError):
                self.store.create_project_with_imported_conversation(
                    "owner", "Rollback", "source-session", [{"role": "user", "content": "hello"}]
                )

        self.assertEqual(self.store.list_projects("owner"), [])

    def test_duplicate_project_names_remain_distinct_by_immutable_id(self) -> None:
        first = self.store.create_project("owner", "Duplicate")
        second, _ = self.store.create_project_with_imported_conversation(
            "owner", "Duplicate", "source-session", []
        )

        self.assertNotEqual(first["id"], second["id"])

    def test_memory_search_and_conflict_history(self) -> None:
        project = self.store.create_project("owner", "Memory")
        old = self.store.add_memory("owner", project["id"], "decision", "Test pressure is 12 bar")
        new = self.store.add_memory(
            "owner", project["id"], "decision", "Test pressure is 15 bar", supersedes=[old["id"]]
        )
        memories = self.store.list_memories("owner", project["id"])
        self.assertEqual({item["id"] for item in memories if item["active"]}, {new["id"]})
        self.assertEqual(next(item for item in memories if item["id"] == old["id"])["superseded_by"], new["id"])
        self.assertEqual(self.store.search("owner", project["id"], "15 bar")["memories"][0]["id"], new["id"])

    def test_file_and_artifact_persistence_are_confined(self) -> None:
        project = self.store.create_project("owner", "Files")
        conversation = self.store.create_conversation("owner", project["id"])
        artifact = self.store.save_file(
            "owner", project["id"], "report.md", b"# Result\n12 bar", "text/markdown", "Result 12 bar",
            conversation["id"], artifact=True, creator="assistant",
        )
        metadata, content = self.store.read_file("owner", project["id"], artifact["id"])
        self.assertEqual(content, b"# Result\n12 bar")
        self.assertEqual(metadata["index_status"], "indexed")
        listed_metadata = self.store.list_file_metadata("owner", project["id"])
        self.assertEqual(listed_metadata[0]["original_name"], "report.md")
        self.assertEqual(listed_metadata[0]["size"], len(content))
        self.assertTrue(self.store.list_files("owner", project["id"])[0]["artifact_id"])
        self.assertEqual(self.store.search("owner", project["id"], "Result")["files"][0]["file_id"], artifact["id"])
        tools = ProjectTools(self.store)
        moved = tools.project_file_move(
            "owner", project["id"], artifact["id"], f"artifacts/reports/{artifact['id']}.md"
        )
        self.assertEqual(moved["storage_path"], f"artifacts/reports/{artifact['id']}.md")
        with self.assertRaises(ProjectPathError):
            tools.project_file_move("owner", project["id"], artifact["id"], "outside.md")
        for unsafe in ("../outside", "/etc/passwd", "files/../../outside"):
            with self.subTest(path=unsafe), self.assertRaises(ProjectPathError):
                self.store.confined_path(project["id"], unsafe)
        files = self.data_root / "projects" / project["id"] / "files"
        (files / "escape").symlink_to(Path(self.temporary.name))
        with self.assertRaises(ProjectPathError):
            self.store.confined_path(project["id"], "files/escape/data.txt")

        project_root = self.data_root / "projects" / project["id"]
        shutil_target = Path(self.temporary.name) / "outside-project"
        shutil_target.mkdir()
        shutil.rmtree(project_root)
        project_root.symlink_to(shutil_target, target_is_directory=True)
        with self.assertRaises(ProjectPathError):
            tools.project_file_create("owner", project["id"], "escaped.txt", "must stay confined")

    def test_storage_offline_blocks_writes_without_fallback(self) -> None:
        offline = ProjectStore(
            Path(self.temporary.name) / "offline.db",
            Path(self.temporary.name) / "missing-mount",
            require_mount=True,
        )
        self.assertFalse(offline.storage_status().online)
        with self.assertRaises(ProjectStorageOfflineError):
            offline.create_project("owner", "Must not fall back")
        self.assertFalse((Path(self.temporary.name) / "missing-mount").exists())

    def test_memory_writer_normalizes_model_confidence(self) -> None:
        self.assertEqual(self.store._normalize_confidence(1.0), "HIGH")
        self.assertEqual(self.store._normalize_confidence(0.6), "MEDIUM")
        self.assertEqual(self.store._normalize_confidence("low"), "LOW")
        self.assertEqual(self.store._normalize_confidence("unexpected"), "MEDIUM")

    def test_runtime_registry_searches_only_the_scoped_project(self) -> None:
        project = self.store.create_project("owner", "Scoped tools")
        other = self.store.create_project("owner", "Other project")
        self.store.add_memory("owner", project["id"], "fact", "Pressure is 12 bar")
        self.store.add_memory("owner", other["id"], "fact", "Pressure is 99 bar")
        scope = ProjectToolScope(ProjectTools(self.store), "owner", project["id"])

        with patch.dict(os.environ, {"MCP_ENABLED": "true", "MCP_PROJECT_ENABLED": "true"}, clear=False):
            results = run_agent_tools("main", "pressure", allow_local_tools=False, project_scope=scope)

        self.assertEqual([result["name"] for result in results], ["project_context"])
        output = __import__("json").loads(results[0]["output"])
        memories = output["context"]["memories"]
        self.assertEqual([memory["content"] for memory in memories], ["Pressure is 12 bar"])
        self.assertEqual(results[0]["details"]["execution"], "mcp")


if __name__ == "__main__":
    unittest.main()
