from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import call, patch

from mcp_servers.google_server import GoogleToolScope
from runtime.agent_runtime import AgentRuntime, LatencyRecorder
from runtime.mcp_host import MCPCallOutcome
from runtime.tool_registry import ProjectToolScope


DATASET = {
    "title": "Supplier capacity",
    "columns": ["Supplier", "Capacity", "Evidence"],
    "column_types": ["text", "integer", "text"],
    "units": [None, "units", None],
    "sources": ["S1"],
    "rows": [["A", 10, "S1"], ["B", 14, "S1"]],
    "chart_candidates": [{
        "title": "Capacity by supplier", "chart_type": "BAR",
        "category_column": 0, "series_columns": [1],
    }],
}


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {
            "choices": [{"message": {"role": "assistant", "content": self.content}, "finish_reason": "stop"}],
            "usage": {},
        }


class SequencedClient:
    def __init__(self, contents: list[str]) -> None:
        self.contents = contents
        self.requests: list[dict[str, object]] = []

    def post(self, _url: str, json: dict[str, object]) -> FakeResponse:
        self.requests.append(json)
        return FakeResponse(self.contents[len(self.requests) - 1])


def outcome(tool: str, output: dict[str, object], *, success: bool = True, status: str = "AVAILABLE") -> MCPCallOutcome:
    return MCPCallOutcome(success, True, tool, "test-mcp", status, output, None if success else status, 1)


class Phase3DOrchestrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.project_scope = ProjectToolScope(SimpleNamespace(), "alice", "project-1", "conversation-1")
        self.google_scope = GoogleToolScope("alice", SimpleNamespace())

    def test_research_plan_accepts_only_known_requested_outputs(self) -> None:
        plan = AgentRuntime._parse_research_plan(
            '{"search_mode":"DEEP_RESEARCH","search_queries":["market"],'
            '"requested_outputs":["google_docs_create","unknown","google_docs_create"]}'
        )

        self.assertEqual(plan.requested_outputs, ("google_docs_create",))

    def test_no_requested_output_skips_planning_and_writes(self) -> None:
        client = SequencedClient([])
        runtime = AgentRuntime(client=client)

        with patch("runtime.agent_runtime.call_mcp_tool") as tool_call:
            report, payload, activity, orchestration = runtime._run_post_research_orchestration(
                "Research this", "Final report", {"usage": {}}, LatencyRecorder(), (),
                self.project_scope, self.google_scope,
            )

        self.assertEqual(report, "Final report")
        self.assertEqual(payload, {"usage": {}})
        self.assertEqual(activity, [])
        self.assertEqual(orchestration["status"], "NOT_REQUESTED")
        self.assertEqual(client.requests, [])
        tool_call.assert_not_called()

    def test_full_chain_injects_report_and_exact_sheet_id_with_user_scopes(self) -> None:
        client = SequencedClient([
            json.dumps({"steps": [
                {"id": "project", "tool": "project_save_artifact", "depends_on": [], "arguments": {"name": "market-report.md"}},
                {"id": "doc", "tool": "google_docs_create", "depends_on": [], "arguments": {"title": "Market report"}},
                {"id": "sheet", "tool": "google_sheets_create", "depends_on": [], "arguments": {"title": "Metrics", "dataset": DATASET}},
                {"id": "chart", "tool": "google_sheets_add_chart", "depends_on": ["sheet"], "arguments": {}},
            ]}),
            "Research complete. Project, Doc, Sheet, and chart were created.",
        ])
        outputs = [
            outcome("project_save_artifact", {"status": "AVAILABLE", "artifact": {"artifact_id": "artifact-1"}}),
            outcome("google_docs_create", {"status": "AVAILABLE", "document_id": "doc-1", "url": "https://docs.test/doc-1"}),
            outcome("google_sheets_create", {"status": "AVAILABLE", "spreadsheet_id": "sheet-1", "url": "https://sheets.test/sheet-1"}),
            outcome("google_sheets_add_chart", {"status": "AVAILABLE", "spreadsheet_id": "sheet-1", "chart_id": 42}),
        ]

        with patch("runtime.agent_runtime.call_mcp_tool", side_effect=outputs) as tool_call:
            answer, _, activity, orchestration = AgentRuntime(client=client)._run_post_research_orchestration(
                "Research and create all outputs", "Final researched evidence", {}, LatencyRecorder(),
                ("project_save_artifact", "google_docs_create", "google_sheets_create", "google_sheets_add_chart"),
                self.project_scope, self.google_scope,
            )

        self.assertTrue(answer.startswith("Final researched evidence"))
        self.assertIn("## 생성된 산출물", answer)
        self.assertIn("https://docs.test/doc-1", answer)
        self.assertIn("https://sheets.test/sheet-1", answer)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(orchestration["status"], "AVAILABLE")
        self.assertEqual([item["name"] for item in activity], [
            "project_save_artifact", "google_docs_create", "google_sheets_create", "google_sheets_add_chart",
        ])
        project_arguments = tool_call.call_args_list[0].args[1]
        doc_arguments = tool_call.call_args_list[1].args[1]
        sheet_arguments = tool_call.call_args_list[2].args[1]
        chart_arguments = tool_call.call_args_list[3].args[1]
        self.assertEqual(project_arguments["content"], "Final researched evidence")
        self.assertEqual(doc_arguments["content"], "Final researched evidence")
        self.assertEqual(sheet_arguments["dataset"], DATASET)
        self.assertEqual(chart_arguments["spreadsheet_id"], "sheet-1")
        self.assertEqual(chart_arguments["data_range"], "A1:B3")
        self.assertEqual(chart_arguments["title"], "Capacity by supplier")
        self.assertIs(tool_call.call_args_list[0].args[2], self.project_scope)
        self.assertIs(tool_call.call_args_list[1].args[4], self.google_scope)
        self.assertEqual(orchestration["steps"][0]["artifact_ids"], ["artifact-1"])
        self.assertEqual(orchestration["steps"][0]["external_urls"], [])
        self.assertEqual(orchestration["steps"][1]["external_urls"], ["https://docs.test/doc-1"])

    def test_failed_sheet_blocks_chart_without_repeating_successful_writes(self) -> None:
        client = SequencedClient([
            json.dumps({"steps": [
                {"id": "doc", "tool": "google_docs_create", "depends_on": [], "arguments": {"title": "Report"}},
                {"id": "sheet", "tool": "google_sheets_create", "depends_on": [], "arguments": {"title": "Metrics", "dataset": DATASET}},
                {"id": "chart", "tool": "google_sheets_add_chart", "depends_on": ["sheet"], "arguments": {}},
            ]}),
            "The Doc succeeded; Sheet and chart need retry.",
        ])
        outputs = [
            outcome("google_docs_create", {"status": "AVAILABLE", "document_id": "doc-1"}),
            outcome("google_sheets_create", {}, success=False, status="SPREADSHEET_CREATE_FAILED"),
        ]

        with patch("runtime.agent_runtime.call_mcp_tool", side_effect=outputs) as tool_call:
            _, _, activity, orchestration = AgentRuntime(client=client)._run_post_research_orchestration(
                "Create a Doc, Sheet, and chart", "Final report", {}, LatencyRecorder(),
                ("google_docs_create", "google_sheets_create", "google_sheets_add_chart"),
                None, self.google_scope,
            )

        self.assertEqual(tool_call.call_count, 2)
        self.assertEqual([item["status"] for item in orchestration["steps"]], [
            "AVAILABLE", "SPREADSHEET_CREATE_FAILED", "DEPENDENCY_FAILED",
        ])
        self.assertEqual(orchestration["status"], "PARTIAL_SUCCESS")
        self.assertEqual(activity[2]["error"], "DEPENDENCY_FAILED")

    def test_artifact_content_removes_operational_tool_messages(self) -> None:
        client = SequencedClient([
            json.dumps({"steps": [
                {"id": "project", "tool": "project_save_artifact", "depends_on": [], "arguments": {"name": "report.md"}},
                {"id": "doc", "tool": "google_docs_create", "depends_on": [], "arguments": {"title": "Report"}},
            ]}),
            "Created.",
        ])
        report = "# Evidence\nVerified fact.\nGoogle Docs를 만들 수 없습니다.\n## Conclusion\nSupported."
        with patch("runtime.agent_runtime.call_mcp_tool", side_effect=[
            outcome("project_save_artifact", {"status": "AVAILABLE"}),
            outcome("google_docs_create", {"status": "AVAILABLE"}),
        ]) as tool_call:
            AgentRuntime(client=client)._run_post_research_orchestration(
                "Create artifacts", report, {}, LatencyRecorder(),
                ("project_save_artifact", "google_docs_create"), self.project_scope, self.google_scope,
            )
        for call_item in tool_call.call_args_list:
            self.assertNotIn("만들 수 없습니다", call_item.args[1]["content"])
            self.assertIn("Verified fact", call_item.args[1]["content"])

    def test_text_only_dataset_creates_sheet_but_skips_chart(self) -> None:
        text_dataset = {
            "title": "Supplier status",
            "columns": ["Supplier", "Status", "Evidence"],
            "column_types": ["text", "text", "text"],
            "units": [None, None, None],
            "sources": ["S1"],
            "rows": [["A", "Likely", "S1"], ["B", "UNKNOWN", "S1"]],
            "chart_candidates": [{
                "title": "Status", "chart_type": "BAR", "category_column": 0, "series_columns": [1],
            }],
        }
        client = SequencedClient([
            json.dumps({"steps": [
                {"id": "sheet", "tool": "google_sheets_create", "depends_on": [], "arguments": {"title": "Status", "dataset": text_dataset}},
                {"id": "chart", "tool": "google_sheets_add_chart", "depends_on": ["sheet"], "arguments": {}},
            ]}),
            "Sheet created; chart skipped.",
        ])
        with patch("runtime.agent_runtime.call_mcp_tool", return_value=outcome(
            "google_sheets_create", {"status": "AVAILABLE", "spreadsheet_id": "sheet-1"},
        )) as tool_call:
            _, _, _, orchestration = AgentRuntime(client=client)._run_post_research_orchestration(
                "Create a Sheet and chart", "Report", {}, LatencyRecorder(),
                ("google_sheets_create", "google_sheets_add_chart"), None, self.google_scope,
            )
        self.assertEqual(tool_call.call_count, 1)
        self.assertEqual(orchestration["steps"][1]["status"], "NO_VALID_CHART_DATA")
        self.assertFalse(orchestration["steps"][1]["retryable"])

    def test_title_only_sheet_plan_extracts_typed_dataset_separately(self) -> None:
        client = SequencedClient([
            json.dumps({"steps": [{
                "id": "sheet", "tool": "google_sheets_create", "depends_on": [],
                "arguments": {"title": "Metrics"},
            }]}),
            json.dumps({"dataset": DATASET}),
            "Sheet created.",
        ])
        with patch("runtime.agent_runtime.call_mcp_tool", return_value=outcome(
            "google_sheets_create", {"status": "AVAILABLE", "spreadsheet_id": "sheet-1"},
        )) as tool_call:
            _, _, _, orchestration = AgentRuntime(client=client)._run_post_research_orchestration(
                "Create a Sheet", "# Report\nTwo sourced values.", {}, LatencyRecorder(),
                ("google_sheets_create",), None, self.google_scope,
            )
        self.assertEqual(orchestration["status"], "AVAILABLE")
        self.assertEqual(tool_call.call_args.args[1]["dataset"], DATASET)
        extraction_request = client.requests[1]["messages"][1]["content"]
        self.assertIn("output_schema", extraction_request)
        self.assertIn("completed_report", extraction_request)

    def test_invalid_extraction_gets_one_bounded_repair_pass(self) -> None:
        client = SequencedClient([
            json.dumps({"steps": [{
                "id": "sheet", "tool": "google_sheets_create", "depends_on": [],
                "arguments": {"title": "Metrics"},
            }]}),
            json.dumps({"dataset": {**DATASET, "rows": [["A", "10", "S1"], ["B", 14, "S1"]]}}),
            json.dumps({"dataset": DATASET}),
            "Sheet created.",
        ])
        with patch("runtime.agent_runtime.call_mcp_tool", return_value=outcome(
            "google_sheets_create", {"status": "AVAILABLE", "spreadsheet_id": "sheet-1"},
        )) as tool_call:
            _, _, _, orchestration = AgentRuntime(client=client)._run_post_research_orchestration(
                "Create a Sheet", "# Report\nTwo sourced values.", {}, LatencyRecorder(),
                ("google_sheets_create",), None, self.google_scope,
            )
        self.assertEqual(orchestration["status"], "AVAILABLE")
        self.assertEqual(tool_call.call_args.args[1]["dataset"], DATASET)
        self.assertIn("validation_error", client.requests[2]["messages"][1]["content"])

    def test_invalid_or_duplicate_plan_falls_back_to_required_outputs(self) -> None:
        invalid_plans = [
            '{"steps":[{"id":"chart","tool":"google_sheets_add_chart","depends_on":[],"arguments":{"chart_type":"LINE","data_range":"A1:B2"}}]}',
            '{"steps":['
            '{"id":"doc1","tool":"google_docs_create","depends_on":[],"arguments":{"title":"One"}},'
            '{"id":"doc2","tool":"google_docs_create","depends_on":[],"arguments":{"title":"Two"}}]}',
        ]
        for plan in invalid_plans:
            client = SequencedClient([plan, "Required Doc created; chart failure reported."])
            with self.subTest(plan=plan), patch(
                "runtime.agent_runtime.call_mcp_tool",
                return_value=outcome("google_docs_create", {"status": "AVAILABLE", "url": "https://docs.test/fallback"}),
            ) as tool_call:
                _, _, activity, orchestration = AgentRuntime(client=client)._run_post_research_orchestration(
                    "Create outputs", "Final report", {}, LatencyRecorder(),
                    ("google_docs_create", "google_sheets_add_chart"), None, self.google_scope,
                )
                self.assertEqual(tool_call.call_count, 1)
                self.assertEqual(tool_call.call_args.args[0], "google_docs_create")
                self.assertEqual(orchestration["status"], "PARTIAL_SUCCESS")
                self.assertTrue(orchestration["goal_satisfied"])
                self.assertEqual({item["name"] for item in activity}, {
                    "google_docs_create", "google_sheets_add_chart",
                })


if __name__ == "__main__":
    unittest.main()