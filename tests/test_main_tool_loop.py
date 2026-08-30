from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from runtime.agent_runtime import AgentRuntime
from runtime.mcp_host import MCPCallOutcome


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
        self.assertEqual(result.tools[0]["capability"], "documentation")
        self.assertEqual(result.tools[0]["action"], "READ")

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
