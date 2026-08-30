from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from vscode_adapter.app import SYNTHETIC_CONTINUATION, app, metrics


class VSCodeAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        for key in metrics:
            metrics[key] = 0.0 if key.endswith("_total") else 0
        self.requests: list[httpx.Request] = []

    def client(self, handler):
        upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return TestClient(app), upstream

    def test_normal_chat_is_not_modified(self) -> None:
        payload = {"model": "qwen3.8-27b", "messages": [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "hello"},
        ], "temperature": 0.1, "extra_body": {"custom": True}}
        encoded = json.dumps(payload).encode()

        async def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(200, json={"choices": []})

        client, upstream = self.client(handler)
        with patch("vscode_adapter.app.httpx.AsyncClient", return_value=upstream), client:
            response = client.post("/v1/chat/completions", content=encoded, headers={"content-type": "application/json"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.requests[0].content, encoded)
        self.assertEqual(metrics["requests_with_user_role"], 1)
        self.assertEqual(metrics["synthetic_continuation_inserted"], 0)

    def test_normal_agent_with_user_and_tool_result_is_not_modified(self) -> None:
        payload = {"model": "qwen3.8-27b", "messages": [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "inspect files"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function", "function": {"name": "read", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ], "tools": [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]}
        encoded = json.dumps(payload).encode()

        async def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(200, json={"choices": []})

        client, upstream = self.client(handler)
        with patch("vscode_adapter.app.httpx.AsyncClient", return_value=upstream), client:
            client.post("/v1/chat/completions", content=encoded, headers={"content-type": "application/json"})

        self.assertEqual(self.requests[0].content, encoded)

    def test_compacted_agent_appends_minimal_user_after_tool_tail(self) -> None:
        payload = {"model": "qwen3.8-27b", "messages": [
            {"role": "system", "content": "compacted goal"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function", "function": {"name": "read", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": "call_1", "content": "first result"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_2", "type": "function", "function": {"name": "search", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": "call_2", "content": "second result"},
        ], "tool_choice": "auto", "stream": False, "stop": ["END"], "custom_field": {"preserved": True}}

        async def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": "done"}}]})

        client, upstream = self.client(handler)
        with patch("vscode_adapter.app.httpx.AsyncClient", return_value=upstream), client:
            response = client.post("/v1/chat/completions", json=payload)

        forwarded = json.loads(self.requests[0].content)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(forwarded["messages"][:-1], payload["messages"])
        self.assertEqual(forwarded["messages"][-1], {"role": "user", "content": SYNTHETIC_CONTINUATION})
        self.assertEqual({key: value for key, value in forwarded.items() if key != "messages"}, {
            key: value for key, value in payload.items() if key != "messages"
        })
        self.assertEqual(metrics["requests_missing_user_role"], 1)
        self.assertEqual(metrics["synthetic_continuation_inserted"], 1)

    def test_empty_messages_are_forwarded_without_repair(self) -> None:
        payload = {"model": "qwen3.8-27b", "messages": []}

        async def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(400, json={"error": {"message": "No user query found in messages."}})

        client, upstream = self.client(handler)
        with patch("vscode_adapter.app.httpx.AsyncClient", return_value=upstream), client:
            response = client.post("/v1/chat/completions", json=payload)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(json.loads(self.requests[0].content), payload)
        self.assertEqual(metrics["synthetic_continuation_inserted"], 0)
        self.assertEqual(metrics["upstream_400"], 1)

    def test_userless_request_without_tool_continuation_is_not_repaired(self) -> None:
        payload = {"model": "qwen3.8-27b", "messages": [{"role": "system", "content": "policy only"}]}

        async def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(400, json={"error": {"message": "No user query found in messages."}})

        client, upstream = self.client(handler)
        with patch("vscode_adapter.app.httpx.AsyncClient", return_value=upstream), client:
            response = client.post("/v1/chat/completions", json=payload)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(json.loads(self.requests[0].content), payload)
        self.assertEqual(metrics["synthetic_continuation_inserted"], 0)

    def test_streaming_sse_is_passed_through(self) -> None:
        class ChunkStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                for chunk in chunks:
                    yield chunk

        chunks = [
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_file","arguments":""}}]}}]}\n\n',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"path\\":\\"runtime/router.py\\"}"}}]},"finish_reason":"tool_calls"}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        payload = {"model": "qwen3.8-27b", "messages": [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "stream"},
        ], "stream": True, "stream_options": {"include_usage": True}}

        async def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=ChunkStream())

        client, upstream = self.client(handler)
        with patch("vscode_adapter.app.httpx.AsyncClient", return_value=upstream), client:
            with client.stream("POST", "/v1/chat/completions", json=payload) as response:
                received = b"".join(response.iter_bytes())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(received, b"".join(chunks))
        self.assertEqual(json.loads(self.requests[0].content), payload)

    def test_models_and_authorization_are_forwarded(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(200, json={"data": [{"id": "qwen3.8-27b"}]})

        client, upstream = self.client(handler)
        with patch("vscode_adapter.app.httpx.AsyncClient", return_value=upstream), client:
            response = client.get("/v1/models", headers={"authorization": "Bearer local-test"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.requests[0].headers["authorization"], "Bearer local-test")


if __name__ == "__main__":
    unittest.main()
