import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient

from runtime.agent_runtime import AgentRuntime
from runtime.web_search import search
from web import app as web_app
from web.auth import UserStore


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {
            "choices": [{"message": {"content": "verified response"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        }


class FakeClient:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def post(self, url: str, json: dict[str, object]) -> FakeResponse:
        self.requests.append({"url": url, "json": json})
        return FakeResponse()


class SearchDecisionClient(FakeClient):
    def post(self, url: str, json: dict[str, object]) -> FakeResponse:
        self.requests.append({"url": url, "json": json})
        content = "QUICK_SEARCH" if len(self.requests) == 1 else "web-verified answer"
        response = FakeResponse()
        response.json = lambda: {  # type: ignore[method-assign]
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        }
        return response


class BraveResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {"web": {"results": [
            {"title": f"Source {index}", "url": f"https://example.com/{index}", "description": "Snippet"}
            for index in range(10)
        ]}}


class NaverResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {"items": [{"title": "<b>국내 출처</b>", "link": "https://naver.example/result", "description": "<b>요약</b>"}]}


class RedditResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {"data": {"children": [{"data": {"title": "Discussion", "permalink": "/r/example/1", "selftext": "Context"}}]}}


class WebRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake_client = FakeClient()
        self.runtime = AgentRuntime(client=self.fake_client)
        self.temporary_directory = TemporaryDirectory()
        self.previous_user_store = web_app.user_store
        web_app.user_store = UserStore(Path(self.temporary_directory.name) / "users.sqlite3")
        web_app.user_store.create("test-admin", "a-test-password", "admin")

    def tearDown(self) -> None:
        web_app.user_store = self.previous_user_store
        self.temporary_directory.cleanup()

    @staticmethod
    def authenticated_client() -> TestClient:
        client = TestClient(web_app.app)
        response = client.post("/api/login", json={"username": "test-admin", "password": "a-test-password"})
        if response.status_code != 200:
            raise AssertionError("test login failed")
        return client

    def test_auto_routes_and_keeps_short_term_session_history(self) -> None:
        first = self.runtime.chat("현재 repository 구조를 실제로 확인해서 설명해줘.", "auto")
        second = self.runtime.chat("방금 요청한 대상은 무엇이었지?", "auto", first.session_id)

        self.assertEqual(first.route.agent, "coding")
        self.assertEqual(second.session_id, first.session_id)
        self.assertTrue(first.tools)
        self.assertEqual({tool["name"] for tool in first.tools}, {
            "list_files",
            "search_files",
            "read_file",
            "git_status",
            "git_diff",
        })
        messages = self.fake_client.requests[-1]["json"]["messages"]
        self.assertTrue(any(message["content"] == "현재 repository 구조를 실제로 확인해서 설명해줘." for message in messages))

    def test_new_session_does_not_include_previous_short_term_history(self) -> None:
        first = self.runtime.chat("테스트 코드명 ALPHA-71", "main")
        new_session = self.runtime.new_session()
        self.runtime.chat("새 session의 첫 질문", "main", new_session)

        messages = self.fake_client.requests[-1]["json"]["messages"]
        self.assertNotEqual(first.session_id, new_session)
        self.assertFalse(any(message["content"] == "테스트 코드명 ALPHA-71" for message in messages))

    def test_direct_server_mode_runs_only_whitelisted_server_tools(self) -> None:
        result = self.runtime.chat("현재 GPU 상태를 확인해줘.", "server")

        self.assertEqual(result.route.agent, "server")
        self.assertEqual({tool["name"] for tool in result.tools}, {
            "nvidia_smi",
            "systemctl_status_qwen_vllm",
            "journalctl_qwen_vllm",
            "df",
            "free",
        })

    def test_fastapi_chat_uses_runtime_not_a_direct_vllm_proxy(self) -> None:
        previous_runtime = web_app.runtime
        web_app.runtime = self.runtime
        try:
            client = self.authenticated_client()
            response = client.post("/api/chat", json={"message": "안녕", "selected_agent": "main"})
        finally:
            web_app.runtime = previous_runtime

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["content"], "verified response")
        self.assertTrue(payload["activity"]["direct"])
        self.assertEqual(self.fake_client.requests[0]["url"], "http://127.0.0.1:8000/v1/chat/completions")
        self.assertEqual(self.fake_client.requests[0]["json"]["chat_template_kwargs"], {"enable_thinking": False})

    def test_web_ui_assets_are_not_cached(self) -> None:
        client = TestClient(web_app.app)

        index = client.get("/", follow_redirects=False)
        login = client.get("/login")
        script = client.get("/static/app.js")

        self.assertEqual(index.status_code, 303)
        self.assertEqual(login.headers["cache-control"], "no-store")
        self.assertEqual(script.headers["cache-control"], "no-store")

    def test_shared_account_cannot_reuse_another_browser_chat_session(self) -> None:
        first_browser = self.authenticated_client()
        second_browser = self.authenticated_client()

        session_id = first_browser.post("/api/new-session").json()["session_id"]
        response = second_browser.post("/api/chat", json={"message": "안녕", "session_id": session_id})

        self.assertEqual(response.status_code, 403)

    def test_auto_current_fact_routes_to_research_and_reports_missing_search_key(self) -> None:
        runtime = AgentRuntime(client=SearchDecisionClient())
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": ""}):
            result = runtime.chat("오늘 서울 날씨는 어때?", "auto")

        self.assertEqual(result.route.agent, "research")
        self.assertEqual(result.route.search_mode, "QUICK_SEARCH")
        self.assertEqual(result.tools[-1]["name"], "web_search")
        self.assertFalse(result.tools[-1]["success"])
        self.assertIn("BRAVE_SEARCH_API_KEY", result.tools[-1]["error"])

    def test_explicit_live_information_uses_quick_search_before_model_decision(self) -> None:
        runtime = AgentRuntime(client=FakeClient())

        self.assertEqual(runtime._search_mode("지금 KT 롤스터와 T1의 최신 경기 상황을 알려줘"), "QUICK_SEARCH")
        self.assertEqual(runtime._search_mode("현재 repository 구조를 실제로 확인해줘"), "NO_SEARCH")

    def test_source_search_request_always_runs_deep_research(self) -> None:
        runtime = AgentRuntime(client=FakeClient())
        message = "수소 액화 저장 관련 최근 2년간 논문, 세미나, 보고서를 검색해줘"
        with patch("runtime.agent_runtime.run_agent_tools", return_value=[]) as run_tools:
            result = runtime.chat(message, "auto")

        self.assertEqual(result.route.agent, "research")
        self.assertEqual(result.route.search_mode, "DEEP_RESEARCH")
        run_tools.assert_called_once_with("research", message, "DEEP_RESEARCH")

    def test_research_uses_larger_output_budget_and_marks_truncation(self) -> None:
        class TruncatedResponse(FakeResponse):
            def json(self) -> dict[str, object]:
                payload = super().json()
                payload["choices"][0]["finish_reason"] = "length"
                return payload

        class TruncatedClient(FakeClient):
            def post(self, url: str, json: dict[str, object]) -> TruncatedResponse:
                self.requests.append({"url": url, "json": json})
                return TruncatedResponse()

        client = TruncatedClient()
        result = AgentRuntime(client=client).chat("논문을 분석해줘", "research")

        self.assertEqual(client.requests[0]["json"]["max_tokens"], 3072)
        self.assertIn("truncated", result.content)

    def test_brave_search_limits_quick_results(self) -> None:
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "test-key"}), patch("runtime.web_search.httpx.get", return_value=BraveResponse()) as get:
            results = search("latest example", "QUICK_SEARCH")

        self.assertEqual(len(results), 5)
        self.assertEqual(get.call_args.kwargs["params"]["count"], 5)

    def test_korean_deep_research_uses_naver_and_reddit(self) -> None:
        responses = [NaverResponse(), RedditResponse()]
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "", "NAVER_SEARCH_CLIENT_ID": "naver-id", "NAVER_SEARCH_CLIENT_SECRET": "naver-secret"}), patch("runtime.web_search.httpx.get", side_effect=responses) as get:
            results = search("국내 최신 정책 비교", "DEEP_RESEARCH")

        self.assertEqual({result["provider"] for result in results}, {"naver", "reddit"})
        self.assertEqual(get.call_args_list[0].args[0], "https://openapi.naver.com/v1/search/webkr.json")
        self.assertEqual(get.call_args_list[1].args[0], "https://www.reddit.com/search.json")


if __name__ == "__main__":
    unittest.main()