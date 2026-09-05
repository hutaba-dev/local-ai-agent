from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from runtime.agent_runtime import AgentRuntime
from runtime.mcp_host import MCPCallOutcome
from mcp_servers.google_server import GoogleToolScope


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self._payload


class SequencedClient:
    def __init__(self, messages: list[dict[str, object]]) -> None:
        self.messages = messages
        self.requests: list[dict[str, object]] = []

    def post(self, _url: str, json: dict[str, object]) -> FakeResponse:
        self.requests.append(json)
        message = self.messages[len(self.requests) - 1]
        return FakeResponse({"choices": [{"message": message, "finish_reason": "stop"}], "usage": {}})


class MainToolLoopTests(unittest.TestCase):
    def test_coding_role_can_select_current_documentation_without_becoming_a_separate_model(self) -> None:
        client = SequencedClient([
            {"role": "assistant", "content": '{"capabilities":["documentation"]}'},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_docs",
                    "type": "function",
                    "function": {
                        "name": "resolve_library_id",
                        "arguments": '{"library_name":"FastAPI","query":"deprecated APIs"}',
                    },
                }],
            },
            {"role": "assistant", "content": "현재 문서를 확인한 결과입니다."},
        ])
        outcome = MCPCallOutcome(
            True, True, "resolve_library_id", "context7-mcp", "AVAILABLE",
            {"status": "AVAILABLE", "source": "Context7", "text": "/fastapi/fastapi"}, None, 1,
        )
        with patch.dict(os.environ, {"MCP_ENABLED": "true"}, clear=False), patch(
            "runtime.agent_runtime.call_mcp_tool", return_value=outcome
        ):
            result = AgentRuntime(client=client).chat(
                "FastAPI API가 deprecated인지 확인해줘", "coding", allow_local_tools=False
            )

        selector_input = client.requests[0]["messages"][1]["content"]
        self.assertIn('"role": "coder"', selector_input)
        self.assertEqual(result.route.agent, "coding")
        self.assertEqual(result.selected_capabilities, ("documentation",))
        self.assertEqual(result.tools[0]["capability"], "documentation")
        self.assertEqual(result.tools[0]["action"], "READ")
        self.assertEqual(
            {tool["function"]["name"] for tool in client.requests[1]["tools"]},
            {"resolve_library_id", "query_documentation"},
        )

    def test_context7_unavailable_does_not_abort_coding_response(self) -> None:
        client = SequencedClient([
            {"role": "assistant", "content": '{"capabilities":["documentation"]}'},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_docs",
                    "type": "function",
                    "function": {
                        "name": "resolve_library_id",
                        "arguments": '{"library_name":"FastAPI","query":"lifespan"}',
                    },
                }],
            },
            {"role": "assistant", "content": "공식 문서 조회는 실패했지만 현재 코드 기준으로 검토했습니다."},
        ])
        outcome = MCPCallOutcome(
            False, True, "resolve_library_id", "context7-mcp", "DEGRADED",
            None, "MCP tool timed out", 21_000,
        )
        with patch.dict(os.environ, {"MCP_ENABLED": "true"}, clear=False), patch(
            "runtime.agent_runtime.call_mcp_tool", return_value=outcome
        ):
            result = AgentRuntime(client=client).chat(
                "FastAPI lifespan 사용을 검토해줘", "coding", allow_local_tools=False
            )

        self.assertIn("실패했지만", result.content)
        self.assertFalse(result.tools[0]["success"])
        self.assertEqual(result.tools[0]["details"]["status"], "DEGRADED")

    def test_git_history_request_exposes_only_scoped_semantic_git_reads(self) -> None:
        client = SequencedClient([
            {"role": "assistant", "content": '{"capabilities":["git"]}'},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_git",
                    "type": "function",
                    "function": {
                        "name": "git_log",
                        "arguments": '{"limit":10,"relative_path":"runtime/search_providers.py"}',
                    },
                }],
            },
            {"role": "assistant", "content": "SearchRouter 변경 이력을 확인했습니다."},
        ])
        outcome = MCPCallOutcome(
            True, True, "git_log", "developer-mcp", "AVAILABLE",
            {"status": "AVAILABLE", "repository": "local-ai-agent", "output": "070310e refactor"}, None, 1,
        )
        with patch.dict(os.environ, {"MCP_ENABLED": "true"}, clear=False), patch(
            "runtime.agent_runtime.call_mcp_tool", return_value=outcome
        ) as call:
            result = AgentRuntime(client=client).chat(
                "SearchRouter.search() 함수가 최근 어떤 변경을 거쳤는지 Git history를 확인해서 설명해줘. 코드는 수정하지 마.",
                "coding", allow_local_tools=False,
            )

        exposed = {tool["function"]["name"] for tool in client.requests[1]["tools"]}
        self.assertEqual(exposed, {"git_status", "git_diff", "git_log", "git_show", "git_blame", "git_branch_info"})
        self.assertNotIn("execute", exposed)
        self.assertEqual(result.tools[0]["capability"], "git")
        call.assert_called_once()

    def test_github_selection_exposes_only_explicit_remote_read_tools(self) -> None:
        client = SequencedClient([
            {"role": "assistant", "content": '{"capabilities":["github"]}'},
            {"role": "assistant", "content": "원격 GitHub 저장소를 읽을 준비가 됐습니다."},
        ])
        with patch.dict(os.environ, {
            "MCP_ENABLED": "true", "MCP_GITHUB_ENABLED": "true", "GITHUB_PERSONAL_ACCESS_TOKEN": "test-token",
        }, clear=False):
            AgentRuntime(client=client).chat("GitHub의 원격 PR 42 변경 파일을 읽어줘", "coding", allow_local_tools=False)

        exposed = {tool["function"]["name"] for tool in client.requests[1]["tools"]}
        self.assertEqual(exposed, {
            "github_search_code", "github_get_file", "github_read_commits",
            "github_read_issues", "github_get_pull_request", "github_read_releases",
        })
        self.assertTrue(all(tool["function"]["name"].startswith("github_") for tool in client.requests[1]["tools"]))

    def test_browser_selection_exposes_only_public_interaction_tools(self) -> None:
        client = SequencedClient([
            {"role": "assistant", "content": '{"capabilities":["browser"]}'},
            {"role": "assistant", "content": "공개 페이지를 렌더링할 준비가 됐습니다."},
        ])
        with patch.dict(os.environ, {
            "MCP_ENABLED": "true", "MCP_PLAYWRIGHT_ENABLED": "true", "MCP_PLAYWRIGHT_EGRESS_GUARD": "true",
        }, clear=False):
            AgentRuntime(client=client).chat("공개 JavaScript 페이지를 렌더링해서 폼 옵션을 확인해줘", "coding", allow_local_tools=False)

        exposed = {tool["function"]["name"] for tool in client.requests[1]["tools"]}
        self.assertEqual(exposed, {"browse_page", "browse_click", "browse_type", "browse_select"})
        self.assertNotIn("browser_run_code_unsafe", exposed)
        self.assertNotIn("browser_file_upload", exposed)

    def test_academic_selection_exposes_only_semantic_scholarly_tools(self) -> None:
        client = SequencedClient([
            {"role": "assistant", "content": '{"capabilities":["academic"]}'},
            {"role": "assistant", "content": "연구자 근거를 조회할 준비가 됐습니다."},
        ])
        with patch.dict(os.environ, {
            "MCP_ENABLED": "true", "MCP_ACADEMIC_ENABLED": "true",
        }, clear=False):
            AgentRuntime(client=client).chat(
                "연구자의 identity와 source별 citation coverage를 평가해줘", "coding", allow_local_tools=False,
            )

        exposed = {tool["function"]["name"] for tool in client.requests[1]["tools"]}
        self.assertEqual(exposed, {
            "academic_resolve_researcher", "academic_search_publications",
            "academic_get_researcher_evidence", "academic_compare_source_coverage",
        })
        self.assertTrue(all(not name.startswith(("scopus_", "wos_", "openalex_")) for name in exposed))

    def test_qwen_selects_capability_calls_tool_and_uses_observation(self) -> None:
        client = SequencedClient([
            {"role": "assistant", "content": '{"capabilities":["time"]}'},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_current_time", "arguments": '{"timezones":["Asia/Seoul"]}'},
                }],
            },
            {"role": "assistant", "content": "서울의 현재 시각입니다."},
        ])
        outcome = MCPCallOutcome(
            True, True, "get_current_time", "developer-mcp", "AVAILABLE",
            {"status": "AVAILABLE", "times": [{"timezone": "Asia/Seoul", "iso": "2026-08-28T02:00:00+09:00"}]},
            None, 1,
        )
        with patch.dict(os.environ, {"MCP_ENABLED": "true"}, clear=False), patch(
            "runtime.agent_runtime.call_mcp_tool", return_value=outcome
        ) as call:
            result = AgentRuntime(client=client).chat("서울은 지금 몇 시야?", "main")

        self.assertEqual(result.content, "서울의 현재 시각입니다.")
        self.assertEqual(result.tools[0]["capability"], "time")
        self.assertEqual(result.tools[0]["action"], "READ")
        self.assertEqual(len(client.requests[1]["tools"]), 2)
        tool_messages = client.requests[2]["messages"]
        self.assertTrue(any(message.get("role") == "tool" and message.get("tool_call_id") == "call_1" for message in tool_messages))
        call.assert_called_once()

    def test_selector_accepts_structured_name_list_and_ignores_host_workspace_tools(self) -> None:
        selected = AgentRuntime._parse_capability_selection(
            '[{"name":"workspace_search","reason":"locate code"},'
            '{"name":"documentation","reason":"check current docs"}]'
        )

        self.assertEqual(selected, ["workspace_search", "documentation"])

    def test_zero_capability_selection_executes_no_tools(self) -> None:
        client = SequencedClient([
            {"role": "assistant", "content": '{"capabilities":[]}'},
            {"role": "assistant", "content": "Transformer는 attention 기반 신경망 구조입니다."},
        ])
        with patch.dict(os.environ, {"MCP_ENABLED": "true"}, clear=False), patch(
            "runtime.agent_runtime.call_mcp_tool"
        ) as call:
            result = AgentRuntime(client=client).chat("Transformer를 설명해줘", "main")

        self.assertEqual(result.tools, [])
        self.assertNotIn("tools", client.requests[1])
        call.assert_not_called()

    def test_non_google_request_does_not_expose_google_tools(self) -> None:
        client = SequencedClient([
            {"role": "assistant", "content": '{"capabilities":[]}'},
            {"role": "assistant", "content": "TCP는 연결형이고 UDP는 비연결형입니다."},
        ])
        scope = GoogleToolScope("alice", SimpleNamespace())
        with patch.dict(os.environ, {"MCP_ENABLED": "true", "MCP_GOOGLE_ENABLED": "true"}, clear=False), patch(
            "runtime.agent_runtime.call_mcp_tool"
        ) as call:
            result = AgentRuntime(client=client).chat(
                "TCP와 UDP 차이 설명해줘", "main", google_scope=scope
            )

        self.assertEqual(result.selected_capabilities, ())
        self.assertNotIn("tools", client.requests[1])
        call.assert_not_called()

    def test_google_request_exposes_and_calls_existing_drive_tool_with_user_scope(self) -> None:
        client = SequencedClient([
            {"role": "assistant", "content": '{"capabilities":["google"]}'},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_drive",
                "type": "function",
                "function": {"name": "google_drive_list", "arguments": '{"limit":5}'},
            }]},
            {"role": "assistant", "content": "접근 가능한 Drive 파일을 확인했습니다."},
        ])
        scope = GoogleToolScope("alice", SimpleNamespace())
        outcome = MCPCallOutcome(
            True, True, "google_drive_list", "google-mcp", "AVAILABLE",
            {"status": "AVAILABLE", "files": []}, None, 1,
        )
        with patch.dict(os.environ, {"MCP_ENABLED": "true", "MCP_GOOGLE_ENABLED": "true"}, clear=False), patch(
            "runtime.agent_runtime.call_mcp_tool", return_value=outcome
        ) as call:
            result = AgentRuntime(client=client).chat(
                "내 Google Drive에서 AHNBYS가 접근 가능한 파일을 보여줘", "main", google_scope=scope
            )

        exposed = {tool["function"]["name"] for tool in client.requests[1]["tools"]}
        self.assertIn("google_drive_list", exposed)
        self.assertEqual(result.selected_capabilities, ("google",))
        self.assertIs(call.call_args.args[4], scope)

    def test_google_docs_request_calls_existing_write_artifact_tool(self) -> None:
        client = SequencedClient([
            {"role": "assistant", "content": '{"capabilities":["google"]}'},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_docs",
                "type": "function",
                "function": {
                    "name": "google_docs_create",
                    "arguments": '{"title":"Meeting notes","content":"Decisions"}',
                },
            }]},
            {"role": "assistant", "content": "Google Docs 문서를 생성했습니다."},
        ])
        scope = GoogleToolScope("alice", SimpleNamespace())
        outcome = MCPCallOutcome(
            True, True, "google_docs_create", "google-mcp", "AVAILABLE",
            {"status": "AVAILABLE", "document_id": "doc-1"}, None, 1,
        )
        with patch.dict(os.environ, {"MCP_ENABLED": "true", "MCP_GOOGLE_ENABLED": "true"}, clear=False), patch(
            "runtime.agent_runtime.call_mcp_tool", return_value=outcome
        ) as call:
            result = AgentRuntime(client=client).chat(
                "회의 결과를 Google Docs 문서로 만들어줘", "main", google_scope=scope
            )

        self.assertEqual(result.selected_capabilities, ("google",))
        self.assertEqual(result.tools[0]["name"], "google_docs_create")
        self.assertEqual(result.tools[0]["action"], "WRITE_ARTIFACT")
        self.assertIs(call.call_args.args[4], scope)

    def test_project_capability_is_unavailable_without_project_scope(self) -> None:
        client = SequencedClient([
            {"role": "assistant", "content": '{"capabilities":["project"]}'},
            {"role": "assistant", "content": "일반 대화에는 Project 도구를 노출하지 않습니다."},
        ])
        with patch.dict(os.environ, {"MCP_ENABLED": "true", "MCP_PROJECT_ENABLED": "true"}, clear=False), patch(
            "runtime.agent_runtime.call_mcp_tool"
        ) as call:
            result = AgentRuntime(client=client).chat("일반 질문입니다", "main")

        selector_catalog = client.requests[0]["messages"][1]["content"]
        self.assertNotIn('"name": "project"', selector_catalog)
        self.assertNotIn("tools", client.requests[1])
        self.assertEqual(result.tools, [])
        call.assert_not_called()

    def test_selected_capability_is_observable_without_a_tool_call(self) -> None:
        client = SequencedClient([
            {"role": "assistant", "content": '{"capabilities":["documentation"]}'},
            {"role": "assistant", "content": "현재 지식으로 답변합니다."},
        ])

        with patch.dict(os.environ, {"MCP_ENABLED": "true"}, clear=False):
            result = AgentRuntime(client=client).chat("공식 문서 관점에서 설명해줘", "main")

        self.assertEqual(result.selected_capabilities, ("documentation",))
        self.assertEqual(result.tools, [])

    def test_project_scope_lazily_exposes_all_semantic_project_tools(self) -> None:
        client = SequencedClient([
            {"role": "assistant", "content": '{"capabilities":["project"]}'},
            {"role": "assistant", "content": "Project 도구를 선택했습니다."},
        ])
        scope = SimpleNamespace(tools=object(), owner_id="owner", project_id="prj_test", conversation_id=None)
        with patch.dict(os.environ, {"MCP_ENABLED": "true", "MCP_PROJECT_ENABLED": "true"}, clear=False):
            AgentRuntime(client=client).chat("프로젝트 자료를 확인해줘", "main", project_scope=scope)

        exposed = {tool["function"]["name"] for tool in client.requests[1]["tools"]}
        self.assertEqual(exposed, {
            "project_get_context", "project_search", "project_list_files", "project_read_file",
            "project_get_memories", "project_save_memory", "project_list_artifacts", "project_save_artifact",
        })

    def test_python_dictionary_explanation_selects_no_capability(self) -> None:
        client = SequencedClient([
            {"role": "assistant", "content": '{"capabilities":[]}'},
            {"role": "assistant", "content": "dictionary는 key-value이고 list는 순서형 sequence입니다."},
        ])
        with patch.dict(os.environ, {"MCP_ENABLED": "true"}, clear=False), patch(
            "runtime.agent_runtime.call_mcp_tool"
        ) as call:
            result = AgentRuntime(client=client).chat(
                "Python에서 dictionary와 list의 차이를 설명해줘.", "coding", allow_local_tools=False
            )

        self.assertEqual(result.tools, [])
        self.assertNotIn("tools", client.requests[1])
        call.assert_not_called()

    def test_research_request_does_not_enter_main_capability_loop(self) -> None:
        client = SequencedClient([
            {"role": "assistant", "content": '{"search_mode":"NO_SEARCH","ready_to_answer":true}'},
            {"role": "assistant", "content": "분석에는 최신 외부 근거가 필요합니다."},
        ])
        with patch.dict(os.environ, {"MCP_ENABLED": "true"}, clear=False), patch(
            "runtime.agent_runtime.call_mcp_tool"
        ) as call:
            result = AgentRuntime(client=client).chat(
                "NVIDIA 실적 분석", "research", allow_local_tools=False
            )

        self.assertEqual(result.tools, [])
        self.assertTrue(all(request["messages"][0]["role"] == "system" for request in client.requests))
        call.assert_not_called()

    def test_multiple_tool_calls_in_one_turn_are_all_executed(self) -> None:
        client = SequencedClient([
            {"role": "assistant", "content": '{"capabilities":["time"]}'},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "get_current_time", "arguments": "{}"}},
                {"id": "call_2", "type": "function", "function": {"name": "convert_time", "arguments": '{"value":"2026-08-28T02:00:00","from_timezone":"Asia/Seoul","to_timezone":"UTC"}'}},
            ]},
            {"role": "assistant", "content": "두 결과를 확인했습니다."},
        ])
        outcome = MCPCallOutcome(True, True, "tool", "developer-mcp", "AVAILABLE", {"status": "AVAILABLE"}, None, 1)
        with patch.dict(os.environ, {"MCP_ENABLED": "true"}), patch(
            "runtime.agent_runtime.call_mcp_tool", return_value=outcome
        ) as call:
            result = AgentRuntime(client=client).chat("서울 시각을 UTC로 바꿔줘", "main")

        self.assertEqual(call.call_count, 2)
        self.assertEqual(len(result.tools), 2)

    def test_malformed_tool_call_is_rejected_without_aborting_chat(self) -> None:
        client = SequencedClient([
            {"role": "assistant", "content": '{"capabilities":["time"]}'},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_bad", "type": "function", "function": {"name": "get_current_time", "arguments": "not-json"}},
            ]},
            {"role": "assistant", "content": "도구 인수가 잘못되어 일반 답변으로 마무리합니다."},
        ])
        with patch.dict(os.environ, {"MCP_ENABLED": "true"}), patch("runtime.agent_runtime.call_mcp_tool") as call:
            result = AgentRuntime(client=client).chat("현재 시간을 알려줘", "main")

        self.assertIn("마무리", result.content)
        self.assertFalse(result.tools[0]["success"])
        call.assert_not_called()


if __name__ == "__main__":
    unittest.main()
