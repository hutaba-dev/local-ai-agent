import json

from runtime.web_search import _s2_author_queries, _select_s2_author, academic_papers, fetch_sources, s2_get_author, s2_get_author_papers, s2_get_paper, s2_search_author, search, unpaywall_get_oa_location
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient

from runtime.agent_runtime import AgentRuntime
from runtime.tool_registry import ToolResult, _academic_evidence_gaps, _research_tools, _researcher_query, run_agent_tools
from runtime.web_search import fetch_sources, search
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
        content = '{"search_mode":"QUICK_SEARCH","queries":["Seoul weather"],"focus":["current weather"]}' if len(self.requests) == 1 else "web-verified answer"
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


class SourceResponse:
    headers = {"content-type": "text/html; charset=utf-8"}
    is_redirect = False
    text = "<html><head><title>Ignored</title><script>secret()</script></head><body><h1>Evidence</h1><p>Verified source text.</p></body></html>"

    def raise_for_status(self) -> None:
        return None


class OpenAlexResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {"results": [{
            "id": "https://openalex.org/W1",
            "doi": "https://doi.org/10.1000/example",
            "title": "Evidence Paper",
            "publication_date": "2025-01-01",
            "cited_by_count": 12,
            "authorships": [{"author": {"display_name": "Researcher"}}],
            "primary_location": {"source": {"display_name": "Journal"}},
        }]}


class SemanticScholarResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {"name": "Researcher", "paperCount": 10, "citationCount": 200, "hIndex": 8, "affiliations": ["University"], "title": "Evidence Paper", "year": 2025, "externalIds": {"DOI": "10.1000/example"}, "venue": "Journal", "authors": [{"name": "Researcher"}], "abstract": "Verified abstract.", "data": [{
            "authorId": "123",
            "name": "Researcher",
            "paperCount": 10,
            "citationCount": 200,
            "hIndex": 8,
            "affiliations": ["University"],
            "title": "Evidence Paper",
            "year": 2025,
            "citationCount": 12,
            "externalIds": {"DOI": "10.1000/example"},
            "venue": "Journal",
            "authors": [{"name": "Researcher"}],
            "abstract": "Verified abstract.",
        }]}


class UnpaywallResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {
            "doi": "10.1000/example",
            "title": "Evidence Paper",
            "best_oa_location": {
                "url": "https://repository.example/paper",
                "host_type": "repository",
                "license": "cc-by",
                "version": "publishedVersion",
            },
        }


class WebRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake_client = FakeClient()
        self.runtime = AgentRuntime(client=self.fake_client)
        self.temporary_directory = TemporaryDirectory()
        self.previous_user_store = web_app.user_store
        web_app.user_store = UserStore(Path(self.temporary_directory.name) / "users.sqlite3")
        web_app.user_store.create("test-admin", "a-test-password", "admin")
        web_app.user_store.create("test-guest", "a-test-password", "guest")

    def tearDown(self) -> None:
        web_app.user_store = self.previous_user_store
        self.temporary_directory.cleanup()

    @staticmethod
    def authenticated_client(username: str = "test-admin") -> TestClient:
        client = TestClient(web_app.app)
        response = client.post("/api/login", json={"username": username, "password": "a-test-password"})
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
            "df",
            "free",
        })

    def test_server_response_redacts_host_and_ip_identifiers(self) -> None:
        class SensitiveResponse(FakeResponse):
            def json(self) -> dict[str, object]:
                return {
                    "choices": [{"message": {"content": "호스트명: private-host.invalid, API: 127.0.0.1:8000"}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4},
                }

        class SensitiveClient(FakeClient):
            def post(self, url: str, json: dict[str, object]) -> SensitiveResponse:
                self.requests.append({"url": url, "json": json})
                return SensitiveResponse()

        with patch("runtime.agent_runtime.run_agent_tools", return_value=[]):
            result = AgentRuntime(client=SensitiveClient()).chat("서버 IP를 알려줘", "server")

        self.assertNotIn("private-host.invalid", result.content)
        self.assertNotIn("127.0.0.1", result.content)
        self.assertIn("[redacted host]", result.content)
        self.assertIn("[redacted IP]", result.content)

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
        self.assertEqual(payload["activity"]["whole_request_usage"]["llm_call_count"], 1)
        self.assertEqual(payload["activity"]["whole_request_usage"]["input_tokens"], 10)
        self.assertEqual(payload["activity"]["whole_request_usage"]["output_tokens"], 4)
        self.assertIn("end_to_end_tokens_per_second", payload["activity"])
        self.assertEqual(payload["activity"]["final_call"]["purpose"], "response")
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

    def test_guest_cannot_access_server_or_coding_agents(self) -> None:
        previous_runtime = web_app.runtime
        web_app.runtime = self.runtime
        try:
            guest = self.authenticated_client("test-guest")
            agents = guest.get("/api/agents").json()["agents"]
            server = guest.post("/api/chat", json={"message": "GPU 상태를 알려줘", "selected_agent": "server"})
            coding = guest.post("/api/chat", json={"message": "코드를 보여줘", "selected_agent": "coding"})
        finally:
            web_app.runtime = previous_runtime

        self.assertEqual({agent["id"] for agent in agents}, {"auto", "main", "research"})
        self.assertEqual(server.status_code, 403)
        self.assertEqual(coding.status_code, 403)
        self.assertFalse(self.fake_client.requests)

    def test_admin_browser_cannot_access_server_or_auto_server_route(self) -> None:
        previous_runtime = web_app.runtime
        web_app.runtime = self.runtime
        try:
            admin = self.authenticated_client()
            agents = admin.get("/api/agents").json()["agents"]
            direct = admin.post("/api/chat", json={"message": "GPU 상태를 알려줘", "selected_agent": "server"})
            coding = admin.post("/api/chat", json={"message": "코드를 보여줘", "selected_agent": "coding"})
            with patch("runtime.agent_runtime.run_agent_tools") as run_tools:
                automatic = admin.post("/api/chat", json={"message": "서버 GPU 상태를 알려줘"})
        finally:
            web_app.runtime = previous_runtime

        self.assertEqual({agent["id"] for agent in agents}, {"auto", "main", "research"})
        self.assertEqual(direct.status_code, 403)
        self.assertEqual(coding.status_code, 403)
        self.assertEqual(automatic.status_code, 403)
        run_tools.assert_not_called()

    def test_guest_auto_route_cannot_execute_server_tools(self) -> None:
        previous_runtime = web_app.runtime
        web_app.runtime = self.runtime
        try:
            guest = self.authenticated_client("test-guest")
            with patch("runtime.agent_runtime.run_agent_tools") as run_tools:
                response = guest.post("/api/chat", json={"message": "현재 GPU 상태를 알려줘"})
        finally:
            web_app.runtime = previous_runtime

        self.assertEqual(response.status_code, 403)
        run_tools.assert_not_called()

    def test_guest_research_has_no_local_project_tools(self) -> None:
        previous_runtime = web_app.runtime
        web_app.runtime = self.runtime
        try:
            guest = self.authenticated_client("test-guest")
            with patch("runtime.agent_runtime.run_agent_tools", return_value=[]) as run_tools:
                response = guest.post("/api/chat", json={"message": "수소 연구를 요약해줘", "selected_agent": "research"})
        finally:
            web_app.runtime = previous_runtime

        self.assertEqual(response.status_code, 200)
        run_tools.assert_called_once_with("research", ("수소 연구를 요약해줘",), "NO_SEARCH", False)
        self.assertEqual(run_agent_tools("research", "수소 연구를 요약해줘", allow_local_tools=False), [])

    def test_admin_browser_research_has_no_local_project_tools(self) -> None:
        previous_runtime = web_app.runtime
        web_app.runtime = self.runtime
        try:
            admin = self.authenticated_client()
            with patch("runtime.agent_runtime.run_agent_tools", return_value=[]) as run_tools:
                response = admin.post("/api/chat", json={"message": "수소 연구를 요약해줘", "selected_agent": "research"})
        finally:
            web_app.runtime = previous_runtime

        self.assertEqual(response.status_code, 200)
        run_tools.assert_called_once_with("research", ("수소 연구를 요약해줘",), "NO_SEARCH", False)

    def test_guest_legacy_session_is_replaced_before_history_is_used(self) -> None:
        old_session = self.runtime.new_session()
        old_session_data = self.runtime.sessions.get_or_create(old_session)
        self.runtime.sessions.append(old_session_data, "assistant", "private GPU diagnostic")
        previous_runtime = web_app.runtime
        web_app.runtime = self.runtime
        try:
            guest = self.authenticated_client("test-guest")
            response = guest.post("/api/chat", json={"message": "이전 답을 반복해줘", "session_id": old_session})
        finally:
            web_app.runtime = previous_runtime

        self.assertEqual(response.status_code, 200)
        self.assertNotEqual(response.json()["session_id"], old_session)
        messages = self.fake_client.requests[-1]["json"]["messages"]
        self.assertFalse(any(message["content"] == "private GPU diagnostic" for message in messages))

    def test_auto_current_fact_routes_to_research_and_reports_missing_search_key(self) -> None:
        runtime = AgentRuntime(client=SearchDecisionClient())
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": ""}):
            result = runtime.chat("오늘 서울 날씨는 어때?", "auto")

        self.assertEqual(result.route.agent, "research")
        self.assertEqual(result.route.search_mode, "QUICK_SEARCH")
        self.assertEqual(result.tools[-1]["name"], "web_search")
        self.assertFalse(result.tools[-1]["success"])
        self.assertIn("BRAVE_SEARCH_API_KEY", result.tools[-1]["error"])

    def test_model_search_decision_controls_search_mode(self) -> None:
        client = SearchDecisionClient()
        runtime = AgentRuntime(client=client)

        self.assertEqual(runtime._search_mode("현재 repository 구조를 실제로 확인해줘"), "QUICK_SEARCH")
        decision_request = client.requests[0]["json"]
        self.assertEqual(decision_request["max_tokens"], 256)
        self.assertEqual(
            decision_request["messages"][1],
            {"role": "user", "content": "현재 repository 구조를 실제로 확인해줘"},
        )

    def test_model_deep_research_decision_runs_search_with_model_query(self) -> None:
        class DeepResearchDecisionClient(FakeClient):
            def post(self, url: str, json: dict[str, object]) -> FakeResponse:
                self.requests.append({"url": url, "json": json})
                content = '{"search_mode":"DEEP_RESEARCH","queries":["liquefied hydrogen storage papers 2024 2026","liquefied hydrogen storage review"],"focus":["recent papers","reviews"]}' if len(self.requests) == 1 else "web-verified answer"
                response = FakeResponse()
                response.json = lambda: {  # type: ignore[method-assign]
                    "choices": [{"message": {"content": content}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4},
                }
                return response

        runtime = AgentRuntime(client=DeepResearchDecisionClient())
        message = "수소 액화 저장 관련 최근 2년간 논문, 세미나, 보고서를 검색해줘"
        with patch("runtime.agent_runtime.run_agent_tools", return_value=[]) as run_tools:
            result = runtime.chat(message, "auto")

        self.assertEqual(result.route.agent, "research")
        self.assertEqual(result.route.search_mode, "DEEP_RESEARCH")
        run_tools.assert_called_once_with(
            "research",
            (message, "liquefied hydrogen storage papers 2024 2026", "liquefied hydrogen storage review"),
            "DEEP_RESEARCH",
            True,
        )

    def test_deep_research_uses_evidence_analyst_critic_and_revision_passes(self) -> None:
        class PipelineClient(FakeClient):
            def post(self, url: str, json: dict[str, object]) -> FakeResponse:
                self.requests.append({"url": url, "json": json})
                responses = (
                    '{"search_mode":"DEEP_RESEARCH","queries":["researcher papers"]}',
                    "analyst draft",
                    "critic feedback",
                    "final revision",
                )
                response = FakeResponse()
                response.json = lambda: {"choices": [{"message": {"content": responses[len(self.requests) - 1]}}], "usage": {}}  # type: ignore[method-assign]
                return response

        runtime = AgentRuntime(client=PipelineClient())
        tools = [{"name": "semantic_scholar", "success": True, "output": json.dumps({"author": {"name": "Researcher"}, "representative_papers": []}), "error": None}]
        with patch("runtime.agent_runtime.run_agent_tools", return_value=tools):
            result = runtime.chat("연구자 역량을 평가해줘", "auto")

        self.assertEqual(result.content, "final revision")
        self.assertEqual(len(runtime._client.requests), 4)
        analyst_input = runtime._client.requests[1]["json"]["messages"][1]["content"]
        critic_input = runtime._client.requests[2]["json"]["messages"][1]["content"]
        self.assertIn("Evidence Package", analyst_input)
        self.assertIn("Analyst Draft", critic_input)

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

        self.assertEqual(client.requests[0]["json"]["max_tokens"], 4096)
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

    def test_deep_research_fetches_public_html_source_text(self) -> None:
        results = [{"title": "Evidence", "url": "https://example.com/paper", "description": "Snippet"}]
        with patch("runtime.web_search.socket.getaddrinfo", return_value=[(2, 1, 6, "", ("93.184.216.34", 0))]), patch(
            "runtime.web_search.httpx.get", return_value=SourceResponse()
        ):
            sources = fetch_sources(results)

        self.assertEqual(sources[0]["url"], "https://example.com/paper")
        self.assertIn("Verified source text.", sources[0]["text"])
        self.assertNotIn("secret", sources[0]["text"])

    def test_source_fetch_rejects_private_network_urls(self) -> None:
        results = [{"title": "Private", "url": "https://127.0.0.1/private", "description": ""}]

        self.assertEqual(fetch_sources(results), [])

    def test_academic_search_returns_structured_work_metadata(self) -> None:
        with patch("runtime.web_search.httpx.get", return_value=OpenAlexResponse()) as get:
            papers = academic_papers(("liquid hydrogen storage",))

        self.assertEqual(papers[0]["title"], "Evidence Paper")
        self.assertEqual(papers[0]["cited_by_count"], 12)
        self.assertEqual(get.call_args.args[0], "https://api.openalex.org/works")

    def test_semantic_scholar_adapters_normalize_author_and_paper_data(self) -> None:
        with patch("runtime.web_search.httpx.get", return_value=SemanticScholarResponse()):
            candidates = s2_search_author("Researcher University")
            author = s2_get_author("123")
            papers = s2_get_author_papers("123")
            paper = s2_get_paper("paper-id")

        self.assertEqual(candidates[0]["author_id"], "123")
        self.assertEqual(author["h_index"], 8)
        self.assertEqual(papers[0]["doi"], "10.1000/example")
        self.assertEqual(paper["title"], "Evidence Paper")

    def test_semantic_scholar_evidence_retries_shorter_author_query(self) -> None:
        empty = type("EmptyResponse", (), {"status_code": 200, "raise_for_status": lambda self: None, "json": lambda self: {"data": []}})()
        with patch("runtime.web_search.httpx.get", side_effect=[empty, SemanticScholarResponse(), SemanticScholarResponse(), SemanticScholarResponse()]):
            from runtime.web_search import semantic_scholar_evidence

            evidence = semantic_scholar_evidence("Researcher University Mechanical Engineering")

        self.assertIsNotNone(evidence)
        self.assertEqual(evidence["author"]["name"], "Researcher")
        self.assertEqual(evidence["identity_status"], "matched")

    def test_semantic_scholar_evidence_uses_search_title_hint(self) -> None:
        empty = type("EmptyResponse", (), {"status_code": 200, "raise_for_status": lambda self: None, "json": lambda self: {"data": []}})()
        with patch("runtime.web_search.httpx.get", side_effect=[empty, empty, SemanticScholarResponse(), SemanticScholarResponse(), SemanticScholarResponse()]):
            from runtime.web_search import semantic_scholar_evidence

            evidence = semantic_scholar_evidence("한글 이름 소속", ("Researcher University",))

        self.assertIsNotNone(evidence)
        self.assertEqual(evidence["author"]["name"], "Researcher")

    def test_researcher_query_prefers_ascii_planner_alias(self) -> None:
        query = _researcher_query(("안호선 교수 연구 실적", "Ho Seon Ahn Incheon National University"))

        self.assertEqual(query, "Ho Seon Ahn Incheon National University")

    def test_s2_author_queries_extracts_and_reorders_hyphenated_professor_name(self) -> None:
        queries = _s2_author_queries("Incheon University Ho-Sun Ahn professor research papers")

        self.assertEqual(queries, ("Ho Sun Ahn", "Sun Ho Ahn"))

    def test_s2_author_queries_extracts_korean_professor_name(self) -> None:
        self.assertEqual(_s2_author_queries("안호선교수 연구 역량"), ("안호선",))

    def test_s2_author_selection_prefers_full_name_over_initials(self) -> None:
        selected = _select_s2_author(
            [
                {"author_id": "1", "name": "S. Ahn", "affiliations": []},
                {"author_id": "2", "name": "Sun Ho Ahn", "affiliations": []},
            ],
            "Ho Sun Ahn",
            "Incheon University Ho-Sun Ahn professor",
        )

        self.assertEqual(selected["author_id"], "2")

    def test_unpaywall_returns_only_legal_oa_location_when_configured(self) -> None:
        with patch.dict(os.environ, {"UNPAYWALL_EMAIL": "research@example.com"}), patch(
            "runtime.web_search.httpx.get", return_value=UnpaywallResponse()
        ):
            location = unpaywall_get_oa_location("https://doi.org/10.1000/example")

        self.assertEqual(location["oa_url"], "https://repository.example/paper")
        self.assertEqual(location["host_type"], "repository")

    def test_gap_selection_uses_semantic_scholar_for_citation_request_only(self) -> None:
        papers = json.dumps([{"title": "Paper", "doi": "10.1000/example", "cited_by_count": 10}] * 3)
        sources = [
            type("Result", (), {"name": "web_sources", "success": True, "output": json.dumps([{"text": "source"}] * 3)})(),
        ]
        with patch("runtime.tool_registry.semantic_scholar_evidence", return_value=None) as semantic, patch(
            "runtime.tool_registry.unpaywall_oa_locations"
        ) as unpaywall:
            evidence = _academic_evidence_gaps(("researcher citation h-index",), papers, sources)

        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].name, "semantic_scholar")
        self.assertFalse(evidence[0].success)
        semantic.assert_called_once()
        unpaywall.assert_not_called()

    def test_deep_research_runs_s2_fallback_when_openalex_has_no_match(self) -> None:
        with patch(
            "runtime.tool_registry._web_search", return_value=ToolResult("web_search", True, "[]", None, 0)
        ), patch("runtime.tool_registry._web_sources", return_value=ToolResult("web_sources", False, "", "empty", 0)), patch(
            "runtime.tool_registry._academic_papers", return_value=ToolResult("academic_papers", False, "", "empty", 0)
        ), patch(
            "runtime.tool_registry.semantic_scholar_evidence", return_value={"author": {}, "representative_papers": []}
        ) as semantic:
            results = _research_tools(("안호선교수 연구 역량을 평가해줘",), "DEEP_RESEARCH", False)

        self.assertEqual(results[-1].name, "semantic_scholar")
        self.assertTrue(results[-1].success)
        semantic.assert_called_once()

    def test_gap_selection_uses_unpaywall_when_public_source_evidence_is_sparse(self) -> None:
        papers = json.dumps([{"title": "Paper", "doi": "10.1000/example", "cited_by_count": 10}] * 3)
        sources = [
            type("Result", (), {"name": "web_sources", "success": True, "output": json.dumps([{"text": "source"}])})(),
        ]
        with patch("runtime.tool_registry.semantic_scholar_evidence") as semantic, patch(
            "runtime.tool_registry.unpaywall_oa_locations", return_value=[]) as unpaywall:
            evidence = _academic_evidence_gaps(("researcher representative papers",), papers, sources)

        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].name, "unpaywall_oa_location")
        self.assertFalse(evidence[0].success)
        semantic.assert_not_called()
        unpaywall.assert_called_once()

    def test_academic_search_returns_structured_work_metadata(self) -> None:
        with patch("runtime.web_search.httpx.get", return_value=OpenAlexResponse()) as get:
            papers = academic_papers(("liquid hydrogen storage",))

        self.assertEqual(papers[0]["title"], "Evidence Paper")
        self.assertEqual(papers[0]["cited_by_count"], 12)
        self.assertEqual(get.call_args.args[0], "https://api.openalex.org/works")


if __name__ == "__main__":
    unittest.main()