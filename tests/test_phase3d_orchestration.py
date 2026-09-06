from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import call, patch

from mcp_servers.google_server import GoogleToolScope
from runtime.agent_runtime import AgentRuntime, LatencyRecorder
from runtime.mcp_host import MCPCallOutcome
from runtime.tool_registry import ProjectToolScope


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
            '{"steps":['
            '{"id":"project","tool":"project_save_artifact","depends_on":[],"arguments":{"name":"market-report.md"}},'
            '{"id":"doc","tool":"google_docs_create","depends_on":[],"arguments":{"title":"Market report"}},'
            '{"id":"sheet","tool":"google_sheets_create","depends_on":[],"arguments":{"title":"Metrics","values":[["Year","Value"],[2024,10],[2025,14]]}},'
            '{"id":"chart","tool":"google_sheets_add_chart","depends_on":["sheet"],"arguments":{"chart_type":"LINE","data_range":"A1:B3","title":"Trend"}}'
            ']}',
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

        self.assertEqual(answer, "Research complete. Project, Doc, Sheet, and chart were created.")
        self.assertEqual(orchestration["status"], "AVAILABLE")
        self.assertEqual([item["name"] for item in activity], [
            "project_save_artifact", "google_docs_create", "google_sheets_create", "google_sheets_add_chart",
        ])
        project_arguments = tool_call.call_args_list[0].args[1]
        doc_arguments = tool_call.call_args_list[1].args[1]
        chart_arguments = tool_call.call_args_list[3].args[1]
        self.assertEqual(project_arguments["content"], "Final researched evidence")
        self.assertEqual(doc_arguments["content"], "Final researched evidence")
        self.assertEqual(chart_arguments["spreadsheet_id"], "sheet-1")
        self.assertIs(tool_call.call_args_list[0].args[2], self.project_scope)
        self.assertIs(tool_call.call_args_list[1].args[4], self.google_scope)
        self.assertEqual(orchestration["steps"][0]["artifact_ids"], ["artifact-1"])
        self.assertEqual(orchestration["steps"][0]["external_urls"], [])
        self.assertEqual(orchestration["steps"][1]["external_urls"], ["https://docs.test/doc-1"])

    def test_failed_sheet_blocks_chart_without_repeating_successful_writes(self) -> None:
        client = SequencedClient([
            '{"steps":['
            '{"id":"doc","tool":"google_docs_create","depends_on":[],"arguments":{"title":"Report"}},'
            '{"id":"sheet","tool":"google_sheets_create","depends_on":[],"arguments":{"title":"Metrics","values":[["Name","Value"],["A",1]]}},'
            '{"id":"chart","tool":"google_sheets_add_chart","depends_on":["sheet"],"arguments":{"chart_type":"BAR","data_range":"A1:B2"}}'
            ']}',
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

    def test_invalid_or_duplicate_plan_executes_nothing(self) -> None:
        invalid_plans = [
            '{"steps":[{"id":"chart","tool":"google_sheets_add_chart","depends_on":[],"arguments":{"chart_type":"LINE","data_range":"A1:B2"}}]}',
            '{"steps":['
            '{"id":"doc1","tool":"google_docs_create","depends_on":[],"arguments":{"title":"One"}},'
            '{"id":"doc2","tool":"google_docs_create","depends_on":[],"arguments":{"title":"Two"}}]}',
        ]
        for plan in invalid_plans:
            with self.subTest(plan=plan), patch("runtime.agent_runtime.call_mcp_tool") as tool_call:
                _, _, activity, orchestration = AgentRuntime(client=SequencedClient([plan]))._run_post_research_orchestration(
                    "Create outputs", "Final report", {}, LatencyRecorder(),
                    ("google_docs_create", "google_sheets_add_chart"), None, self.google_scope,
                )
                self.assertEqual(activity, [])
                self.assertEqual(orchestration["status"], "NOT_REQUESTED")
                tool_call.assert_not_called()


if __name__ == "__main__":
    unittest.main()