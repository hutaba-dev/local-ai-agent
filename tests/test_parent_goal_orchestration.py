from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mcp_servers.google_server import GoogleToolScope
from runtime.agent_runtime import AgentRuntime, ResearchPlan, TaskGoal
from runtime.mcp_host import MCPCallOutcome


DATASET = {
    "title": "Samsung performance",
    "columns": ["Quarter", "Revenue", "Evidence"],
    "column_types": ["text", "integer", "text"],
    "units": [None, "KRW", None],
    "sources": ["S1"],
    "rows": [["Q1", 100, "S1"], ["Q2", 120, "S1"]],
    "chart_candidates": [],
}
RESEARCH_RESULT = {
    "status": "RESEARCH_COMPLETED",
    "body_markdown": "# Samsung report\nEvidence-backed performance summary. [S1]",
    "sources": [{"id": "S1", "url": "https://example.test/source"}],
    "annotations": ["FACT"],
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


def outcome(tool: str, url: str) -> MCPCallOutcome:
    key = "document_id" if tool == "google_docs_create" else "spreadsheet_id"
    return MCPCallOutcome(
        True, True, tool, "google-mcp", "AVAILABLE",
        {"status": "AVAILABLE", key: f"{tool}-1", "url": url}, None, 1,
    )


def research_state() -> dict[str, object]:
    return {
        "state": "COMPLETE",
        "rounds": [],
        "final_synthesis_executed": True,
        "termination_reason": "llm_evidence_sufficient",
        "result": RESEARCH_RESULT,
    }


class ParentGoalOrchestrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.google_scope = GoogleToolScope("alice", SimpleNamespace())

    def run_research_task(self, outputs: tuple[str, ...]):
        steps = []
        calls = []
        if "google_docs_create" in outputs:
            steps.append({
                "id": "doc", "tool": "google_docs_create", "depends_on": [],
                "arguments": {"title": "Samsung report"},
            })
            calls.append(outcome("google_docs_create", "https://docs.test/report"))
        if "google_sheets_create" in outputs:
            steps.append({
                "id": "sheet", "tool": "google_sheets_create", "depends_on": [],
                "arguments": {"title": "Samsung metrics", "dataset": DATASET},
            })
            calls.append(outcome("google_sheets_create", "https://sheets.test/report"))
        client = SequencedClient([
            json.dumps({"steps": steps}),
            "Completed requested deliverables.",
        ] if outputs else [])
        runtime = AgentRuntime(client=client)
        plan = ResearchPlan(
            "DEEP_RESEARCH", search_queries=("Samsung performance",),
            requested_outputs=outputs, recommended_agent="research",
        )
        with patch.object(runtime, "_search_decision", return_value=plan), patch.object(
            runtime, "_run_deep_research",
            return_value=([], RESEARCH_RESULT["body_markdown"], {}, research_state()),
        ) as research, patch("runtime.agent_runtime.call_mcp_tool", side_effect=calls) as tool_call:
            result = runtime.chat(
                "Research Samsung and create requested outputs", "auto",
                google_scope=self.google_scope,
            )
        return runtime, result, research, tool_call

    def test_research_only_finishes_after_research(self) -> None:
        _, result, research, tool_call = self.run_research_task(())
        research.assert_called_once()
        tool_call.assert_not_called()
        self.assertEqual(result.content, RESEARCH_RESULT["body_markdown"])
        self.assertEqual(result.research["result"]["status"], "RESEARCH_COMPLETED")

    def test_research_then_docs_finishes_with_link(self) -> None:
        _, result, _, tool_call = self.run_research_task(("google_docs_create",))
        self.assertEqual([call.args[0] for call in tool_call.call_args_list], ["google_docs_create"])
        self.assertIn("https://docs.test/report", result.content)
        self.assertTrue(result.orchestration["goal_satisfied"])

    def test_research_then_sheets_finishes_with_link(self) -> None:
        _, result, _, tool_call = self.run_research_task(("google_sheets_create",))
        self.assertEqual([call.args[0] for call in tool_call.call_args_list], ["google_sheets_create"])
        self.assertIn("https://sheets.test/report", result.content)

    def test_research_docs_and_sheets_runs_every_required_output_once(self) -> None:
        _, result, _, tool_call = self.run_research_task(("google_docs_create", "google_sheets_create"))
        self.assertEqual([call.args[0] for call in tool_call.call_args_list], [
            "google_docs_create", "google_sheets_create",
        ])
        self.assertIn("https://docs.test/report", result.content)
        self.assertIn("https://sheets.test/report", result.content)
        self.assertEqual(result.orchestration["events"][-2:], [
            "Goal satisfaction passed", "Final response emitted",
        ])

    def test_parent_recovers_explicit_google_outputs_omitted_by_research_planner(self) -> None:
        client = SequencedClient([
            json.dumps({"steps": [
                {"id": "doc", "tool": "google_docs_create", "depends_on": [], "arguments": {"title": "Report"}},
                {"id": "sheet", "tool": "google_sheets_create", "depends_on": [], "arguments": {"title": "Metrics", "dataset": DATASET}},
            ]}),
            "Completed with links.",
        ])
        runtime = AgentRuntime(client=client)
        planner_without_outputs = ResearchPlan(
            "DEEP_RESEARCH", True, "deep", search_queries=("Samsung performance",),
            requested_outputs=(), recommended_agent="research",
        )
        report = RESEARCH_RESULT["body_markdown"] + "\n이 Research 세션에서는 Docs/Sheets 기능이 없습니다."
        with patch.object(runtime, "_search_decision", return_value=planner_without_outputs), patch.object(
            runtime, "_run_deep_research", return_value=([], report, {}, research_state()),
        ), patch("runtime.agent_runtime.call_mcp_tool", side_effect=[
            outcome("google_docs_create", "https://docs.test/recovered"),
            outcome("google_sheets_create", "https://sheets.test/recovered"),
        ]) as tool_call:
            result = runtime.chat(
                "삼성전자를 조사해 Google Docs와 Sheets로 만들고 링크를 줘", "auto",
                google_scope=self.google_scope,
            )

        self.assertEqual([call.args[0] for call in tool_call.call_args_list], [
            "google_docs_create", "google_sheets_create",
        ])
        self.assertEqual(result.orchestration["required_deliverables"], [
            "google_docs_create", "google_sheets_create",
        ])
        self.assertNotIn("Research 세션에서는", tool_call.call_args_list[0].args[1]["content"])
        self.assertIn("https://docs.test/recovered", result.content)
        self.assertIn("https://sheets.test/recovered", result.content)

    def test_parent_does_not_infer_google_outputs_from_generic_document_language(self) -> None:
        plan = ResearchPlan("DEEP_RESEARCH", True, "deep", search_queries=("topic",))

        goal = TaskGoal.from_request(plan, "자료를 조사해서 보고서 문서로 정리해줘")

        self.assertEqual(goal.deliverables, ())

    def test_completed_google_writes_persist_and_are_not_repeated(self) -> None:
        runtime, first, _, first_tools = self.run_research_task(("google_docs_create", "google_sheets_create"))
        runtime._client.contents.append("Existing Google artifacts reused.")
        followup_plan = ResearchPlan(
            "NO_SEARCH", requested_outputs=("google_docs_create", "google_sheets_create"), recommended_agent="main",
        )
        with patch.object(runtime, "_search_decision", return_value=followup_plan), patch(
            "runtime.agent_runtime.call_mcp_tool",
        ) as repeated_write:
            second = runtime.chat(
                "방금 만든 Google Docs와 Sheets 링크를 다시 줘", "auto", first.session_id,
                google_scope=self.google_scope,
            )

        self.assertEqual(first_tools.call_count, 2)
        repeated_write.assert_not_called()
        self.assertEqual(second.route.agent, "main")
        self.assertEqual(second.research["termination_reason"], "reused_completed_research")
        self.assertIn("https://docs.test/report", second.content)
        self.assertIn("https://sheets.test/report", second.content)

    def test_followup_docs_reuses_research_without_research_rerun(self) -> None:
        client = SequencedClient([
            json.dumps({"steps": [{
                "id": "doc", "tool": "google_docs_create", "depends_on": [],
                "arguments": {"title": "Prior report"},
            }]}),
            "Created from the prior report.",
        ])
        runtime = AgentRuntime(client=client)
        session = runtime.sessions.create()
        runtime.sessions.append(session, "assistant", RESEARCH_RESULT["body_markdown"], {
            "research_result": RESEARCH_RESULT,
        })
        plan = ResearchPlan("NO_SEARCH", requested_outputs=("google_docs_create",), recommended_agent="main")
        with patch.object(runtime, "_search_decision", return_value=plan), patch.object(
            runtime, "_run_deep_research",
        ) as research, patch(
            "runtime.agent_runtime.call_mcp_tool",
            return_value=outcome("google_docs_create", "https://docs.test/followup"),
        ) as tool_call:
            result = runtime.chat("그 보고서를 Google Docs로 만들어줘", "auto", session.id, google_scope=self.google_scope)

        research.assert_not_called()
        self.assertEqual(result.session_id, session.id)
        self.assertEqual(result.route.agent, "main")
        self.assertEqual(tool_call.call_args.args[1]["content"], RESEARCH_RESULT["body_markdown"])
        self.assertIn("https://docs.test/followup", result.content)

    def test_followup_sheets_reuses_research_without_research_rerun(self) -> None:
        client = SequencedClient([
            json.dumps({"steps": [{
                "id": "sheet", "tool": "google_sheets_create", "depends_on": [],
                "arguments": {"title": "Prior metrics", "dataset": DATASET},
            }]}),
            "Created from prior metrics.",
        ])
        runtime = AgentRuntime(client=client)
        session = runtime.sessions.create()
        runtime.sessions.append(session, "assistant", RESEARCH_RESULT["body_markdown"], {
            "research_result": RESEARCH_RESULT,
        })
        plan = ResearchPlan("NO_SEARCH", requested_outputs=("google_sheets_create",), recommended_agent="main")
        with patch.object(runtime, "_search_decision", return_value=plan), patch.object(
            runtime, "_run_deep_research",
        ) as research, patch(
            "runtime.agent_runtime.call_mcp_tool",
            return_value=outcome("google_sheets_create", "https://sheets.test/followup"),
        ):
            result = runtime.chat("표로 정리해서 Sheets로 만들어", "auto", session.id, google_scope=self.google_scope)
        research.assert_not_called()
        self.assertIn("https://sheets.test/followup", result.content)

    def test_completed_doc_is_not_written_twice(self) -> None:
        client = SequencedClient(["Existing document reused."])
        runtime = AgentRuntime(client=client)
        session = runtime.sessions.create()
        prior = {
            "status": "AVAILABLE",
            "steps": [{
                "step_id": "doc", "tool": "google_docs_create", "status": "AVAILABLE",
                "artifact_ids": ["doc-1"], "external_urls": ["https://docs.test/existing"],
                "error_category": None, "retryable": False,
            }],
        }
        runtime.sessions.append(session, "assistant", RESEARCH_RESULT["body_markdown"], {
            "research_result": RESEARCH_RESULT, "orchestration": prior,
        })
        plan = ResearchPlan("NO_SEARCH", requested_outputs=("google_docs_create",), recommended_agent="main")
        with patch.object(runtime, "_search_decision", return_value=plan), patch(
            "runtime.agent_runtime.call_mcp_tool",
        ) as tool_call:
            result = runtime.chat("그 Docs 링크 다시 줘", "auto", session.id, google_scope=self.google_scope)
        tool_call.assert_not_called()
        self.assertIn("https://docs.test/existing", result.content)

    def test_unavailable_required_capability_is_explicit_failure(self) -> None:
        runtime = AgentRuntime(client=SequencedClient(["Google Docs is unavailable."]))
        _, _, activity, orchestration = runtime._run_post_research_orchestration(
            "Create a Doc", RESEARCH_RESULT["body_markdown"], {},
            __import__("runtime.agent_runtime", fromlist=["LatencyRecorder"]).LatencyRecorder(),
            ("google_docs_create",), None, None,
        )
        self.assertEqual(activity[0]["error"], "CAPABILITY_UNAVAILABLE")
        self.assertEqual(orchestration["status"], "FAILED")
        self.assertTrue(orchestration["goal_satisfied"])


if __name__ == "__main__":
    unittest.main()
