from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mcp_servers.google_server import GoogleToolScope
from runtime.agent_runtime import AgentRuntime, LatencyRecorder, ResearchPlan
from runtime.mcp_host import MCPCallOutcome
from runtime.tool_registry import ProjectToolScope


VISUAL_BRIEF = {
    "purpose": "Explain the evidence-backed causal relationship",
    "title": "AI demand to HBM impact",
    "key_entities": ["AI data centers", "GPU roadmap", "HBM suppliers"],
    "relationships": ["AI demand increases accelerator demand", "Accelerators increase HBM demand"],
    "hierarchy": ["Demand", "Compute", "Memory impact"],
    "key_numbers": ["6-12 month outlook"],
    "visual_style": "clean editorial concept diagram",
    "constraints": ["minimal text", "no invented logos"],
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


def plan(*steps: dict[str, object]) -> str:
    return json.dumps({"steps": list(steps)})


def step(step_id: str, tool: str, arguments: dict[str, object], dependencies: list[str] | None = None) -> dict[str, object]:
    return {"id": step_id, "tool": tool, "depends_on": dependencies or [], "arguments": arguments}


def outcome(
    tool: str, output: dict[str, object] | None, *, success: bool = True, status: str = "AVAILABLE",
) -> MCPCallOutcome:
    return MCPCallOutcome(success, True, tool, "test-mcp", status, output, None if success else status, 1)


class Phase3EOrchestrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.project_scope = ProjectToolScope(SimpleNamespace(), "alice", "project-1", "conversation-1")
        self.google_scope = GoogleToolScope("alice", SimpleNamespace())

    def run_orchestration(
        self,
        planner: str,
        outputs: tuple[str, ...],
        outcomes: list[MCPCallOutcome],
        *,
        project: bool = False,
        google: bool = False,
        report: str = "FINAL_REPORT_ONLY",
    ):
        client = SequencedClient([planner, "Final response with all operational results."])
        with patch("runtime.agent_runtime.call_mcp_tool", side_effect=outcomes) as tool_call:
            result = AgentRuntime(client=client)._run_post_research_orchestration(
                "Research and create requested deliverables", report, {}, LatencyRecorder(), outputs,
                self.project_scope if project else None,
                self.google_scope if google else None,
                "session-owner",
            )
        return client, tool_call, result

    def test_research_to_media_uses_only_compact_visual_brief(self) -> None:
        media_output = {"status": "AVAILABLE", "result": {
            "job_id": "mjob-1", "image_id": "img-1", "artifact_id": None,
        }}
        client, tool_call, (_, _, activity, orchestration) = self.run_orchestration(
            plan(step("visual", "media_generate_image", {"visual_brief": VISUAL_BRIEF})),
            ("media_generate_image",), [outcome("media_generate_image", media_output)],
            report="RAW_SOURCE_AND_FULL_REPORT_MUST_NOT_REAPPEAR",
        )

        arguments = tool_call.call_args.args[1]
        self.assertEqual(tool_call.call_args.args[3], "session-owner")
        self.assertFalse(arguments["save_to_project"])
        self.assertIn("AI data centers", arguments["intent"])
        self.assertNotIn("RAW_SOURCE_AND_FULL_REPORT", json.dumps(arguments))
        self.assertEqual(activity[0]["capability"], "media")
        self.assertEqual(orchestration["steps"][0]["artifact_ids"], ["img-1", "mjob-1"])
        self.assertIn("RAW_SOURCE_AND_FULL_REPORT", client.requests[0]["messages"][1]["content"])

    def test_research_project_media_saves_nested_image_artifact_with_identity(self) -> None:
        media_output = {"status": "AVAILABLE", "result": {
            "job_id": "mjob-2", "image_id": "img-2", "artifact_id": "art-image-2",
            "worker": "ahn7", "model": "stabilityai/sd-turbo", "operation": "generate",
        }}
        _, tool_call, (_, _, _, orchestration) = self.run_orchestration(
            plan(
                step("report", "project_save_artifact", {"name": "report.md"}),
                step("visual", "media_generate_image", {"visual_brief": VISUAL_BRIEF}),
            ),
            ("project_save_artifact", "media_generate_image"),
            [
                outcome("project_save_artifact", {"status": "AVAILABLE", "artifact": {"artifact_id": "art-report"}}),
                outcome("media_generate_image", media_output),
            ],
            project=True,
        )

        media_arguments = tool_call.call_args_list[1].args[1]
        self.assertTrue(media_arguments["save_to_project"])
        self.assertEqual(media_arguments["source_references"], ["research.final_synthesis"])
        self.assertEqual(media_arguments["related_results"]["report_result"], "steps.report.result")
        self.assertIs(tool_call.call_args_list[1].args[2], self.project_scope)
        self.assertEqual(self.project_scope.owner_id, "alice")
        self.assertEqual(self.project_scope.project_id, "project-1")
        self.assertEqual(self.project_scope.conversation_id, "conversation-1")
        self.assertIn("art-image-2", orchestration["steps"][1]["artifact_ids"])

    def test_docs_sheets_chart_media_full_chain_preserves_phase3d_dependencies(self) -> None:
        planner = plan(
            step("doc", "google_docs_create", {"title": "Report"}),
            step("sheet", "google_sheets_create", {"title": "Comparison", "values": [["Company", "Score"], ["A", 1]]}),
            step("chart", "google_sheets_add_chart", {"chart_type": "BAR", "data_range": "A1:B2"}, ["sheet"]),
            step("visual", "media_generate_image", {"visual_brief": VISUAL_BRIEF}),
        )
        outputs = [
            outcome("google_docs_create", {"status": "AVAILABLE", "document_id": "doc-1", "url": "https://docs.test/1"}),
            outcome("google_sheets_create", {"status": "AVAILABLE", "spreadsheet_id": "sheet-1", "url": "https://sheets.test/1"}),
            outcome("google_sheets_add_chart", {"status": "AVAILABLE", "spreadsheet_id": "sheet-1", "chart_id": 7}),
            outcome("media_generate_image", {"status": "AVAILABLE", "result": {"job_id": "mjob-3", "image_id": "img-3"}}),
        ]

        _, tool_call, (_, _, activity, orchestration) = self.run_orchestration(
            planner,
            ("google_docs_create", "google_sheets_create", "google_sheets_add_chart", "media_generate_image"),
            outputs,
            google=True,
        )

        self.assertEqual([item["name"] for item in activity], [
            "google_docs_create", "google_sheets_create", "google_sheets_add_chart", "media_generate_image",
        ])
        self.assertEqual(tool_call.call_args_list[2].args[1]["spreadsheet_id"], "sheet-1")
        media_refs = tool_call.call_args_list[3].args[1]["related_results"]
        self.assertEqual(media_refs["doc_url"], "https://docs.test/1")
        self.assertEqual(media_refs["sheet_url"], "https://sheets.test/1")
        self.assertEqual(orchestration["status"], "AVAILABLE")

    def test_media_not_requested_is_not_selected_or_executed(self) -> None:
        plan_value = AgentRuntime._parse_research_plan(
            '{"search_mode":"DEEP_RESEARCH","search_queries":["topic"],"requested_outputs":[]}'
        )
        client = SequencedClient([])
        with patch("runtime.agent_runtime.call_mcp_tool") as tool_call:
            _, _, activity, orchestration = AgentRuntime(client=client)._run_post_research_orchestration(
                "Research only", "Report", {}, LatencyRecorder(), plan_value.requested_outputs,
                self.project_scope, self.google_scope, "session-owner",
            )
        self.assertEqual(activity, [])
        self.assertEqual(orchestration["status"], "NOT_REQUESTED")
        tool_call.assert_not_called()

    def test_research_failure_never_runs_media(self) -> None:
        runtime = AgentRuntime(client=SequencedClient([]))
        research_plan = ResearchPlan(
            "DEEP_RESEARCH", search_queries=("topic",), requested_outputs=("media_generate_image",),
            recommended_agent="research",
        )
        with patch.object(runtime, "_search_decision", return_value=research_plan), patch.object(
            runtime, "_run_deep_research", side_effect=ValueError("research failed")
        ), patch("runtime.agent_runtime.call_mcp_tool") as tool_call:
            with self.assertRaisesRegex(ValueError, "research failed"):
                runtime.chat("Research and make a concept image", "research")
        tool_call.assert_not_called()

    def test_media_failure_is_partial_success_and_has_no_gpu_fallback_or_retry(self) -> None:
        _, tool_call, (_, _, activity, orchestration) = self.run_orchestration(
            plan(
                step("report", "project_save_artifact", {"name": "report.md"}),
                step("visual", "media_generate_image", {"visual_brief": VISUAL_BRIEF}),
            ),
            ("project_save_artifact", "media_generate_image"),
            [
                outcome("project_save_artifact", {"status": "AVAILABLE", "artifact": {"artifact_id": "art-report"}}),
                outcome("media_generate_image", None, success=False, status="UNAVAILABLE"),
            ],
            project=True,
        )

        self.assertEqual(tool_call.call_count, 2)
        self.assertEqual(orchestration["status"], "PARTIAL_SUCCESS")
        self.assertEqual(orchestration["steps"][1]["status"], "UNAVAILABLE")
        self.assertTrue(orchestration["steps"][1]["retryable"])
        self.assertEqual(activity[1]["error"], "UNAVAILABLE")

    def test_duplicate_media_plan_is_rejected_before_generation(self) -> None:
        duplicate = plan(
            step("visual_one", "media_generate_image", {"visual_brief": VISUAL_BRIEF}),
            step("visual_two", "media_generate_image", {"visual_brief": VISUAL_BRIEF}),
        )
        with patch("runtime.agent_runtime.call_mcp_tool") as tool_call:
            _, _, activity, orchestration = AgentRuntime(client=SequencedClient([duplicate]))._run_post_research_orchestration(
                "Create one visual", "Report", {}, LatencyRecorder(), ("media_generate_image",),
                self.project_scope, None, "session-owner",
            )
        self.assertEqual(activity, [])
        self.assertEqual(orchestration["status"], "NOT_REQUESTED")
        tool_call.assert_not_called()

    def test_worker_credentials_never_enter_planner_or_media_arguments(self) -> None:
        secret = "phase3e-worker-secret-must-not-leak"
        with patch.dict("os.environ", {"IMAGE_WORKER_TOKEN": secret}, clear=False):
            client, tool_call, _ = self.run_orchestration(
                plan(step("visual", "media_generate_image", {"visual_brief": VISUAL_BRIEF})),
                ("media_generate_image",),
                [outcome("media_generate_image", {"status": "AVAILABLE", "result": {"image_id": "img-safe"}})],
                project=True,
            )
        self.assertNotIn(secret, json.dumps(client.requests))
        self.assertNotIn(secret, json.dumps(tool_call.call_args.args[1]))


if __name__ == "__main__":
    unittest.main()