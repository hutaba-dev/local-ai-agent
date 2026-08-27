import json
import sqlite3

from runtime.web_search import _s2_author_queries, _select_s2_author, academic_papers, fetch_sources, s2_get_author, s2_get_author_papers, s2_get_paper, s2_search_author, search, unpaywall_get_oa_location
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient

from runtime.agent_runtime import AgentRuntime, LatencyRecorder
from runtime.image_client import (
    GeneratedImage,
    ImageEditCompletion,
    ImageEditOperation,
    ImageEditPlan,
    ImagePromptPlan,
    ImageQualityAssessment,
)
from runtime.projects import ProjectStore
from runtime.tool_registry import (
    ToolResult,
    _rank_relevant_web_results,
    _source_queries,
    _academic_evidence_gaps,
    _research_tools,
    _researcher_query,
    research_source_plan,
    run_agent_tools,
)
from runtime.web_search import fetch_sources, search, visual_search
from web import app as web_app
from web.auth import UserStore
from web.uploads import ExtractedUpload


def planned_prompt(request: str, **_kwargs: object) -> ImagePromptPlan:
    return ImagePromptPlan(request, "person", "action", "anime", "high", "high", "core object")


PASSED_IMAGE_QUALITY = ImageQualityAssessment(True, True, (), "all requested elements are readable")


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


class BraveImageResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {"results": [{
            "title": f"Visual {index}",
            "url": f"https://example.com/visual/{index}",
            "thumbnail": {"src": f"https://images.example.com/{index}.jpg"},
        } for index in range(6)]}


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
    text = (
        "<html><head><title>Ignored</title><script>secret()</script></head><body><h1>Evidence</h1><p>"
        + "Verified source text with substantive details for evidence review. " * 5
        + "</p></body></html>"
    )

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
    def test_research_source_router_covers_market_academic_and_mixed_intents(self) -> None:
        cases = {
            "NVIDIA 오늘 실적 발표 몇 시고 시장 예상은 어때?": (
                {"CURRENT_NEWS", "MARKET_FINANCE", "COMPANY_RESEARCH"}, False,
            ),
            "안호선 교수 연구자로서의 역량을 논문 기반으로 평가해줘": (
                {"ACADEMIC_RESEARCH"}, True,
            ),
            "오늘 비트코인 시장 뉴스 정리해줘": (
                {"CURRENT_NEWS", "MARKET_FINANCE"}, False,
            ),
            "AI가 주식시장에 미치는 영향에 대한 학술 논문을 조사해줘": (
                {"ACADEMIC_RESEARCH"}, True,
            ),
            "TSMC 최신 실적과 향후 반도체 수요 전망": (
                {"CURRENT_NEWS", "MARKET_FINANCE", "COMPANY_RESEARCH"}, False,
            ),
            "NVIDIA 실적과 AI bubble에 대한 학계 연구를 함께 비교해줘": (
                {"MARKET_FINANCE", "COMPANY_RESEARCH", "ACADEMIC_RESEARCH", "MIXED"}, True,
            ),
        }
        for query, (expected, academic_enabled) in cases.items():
            with self.subTest(query=query):
                plan = research_source_plan(query)
                self.assertTrue(expected <= set(plan.intents))
                self.assertEqual(plan.academic_enabled, academic_enabled)
                if "CURRENT_NEWS" in expected:
                    self.assertEqual(plan.freshness_priority, "VERY_HIGH")
                if not academic_enabled:
                    self.assertIn("Academic Intelligence — not relevant", plan.skipped_sources)

    def test_current_market_research_forbids_academic_tools_and_fetches_pages(self) -> None:
        queries = (
            "NVIDIA 오늘 실적 발표 일정과 시장 전망, 컨센서스, 주가에 어떤 영향을 줄 것으로 보는지 조사해줘.",
        )
        search_result = ToolResult("web_search", True, json.dumps([{
            "title": "NVIDIA Investor Relations",
            "url": "https://investor.nvidia.com/events-and-presentations/",
            "description": "Earnings and conference call",
            "relevance_score": 1.0,
        }]), None, 1)
        fetched = ToolResult("web_sources", True, json.dumps([{
            "title": "NVIDIA Investor Relations",
            "url": "https://investor.nvidia.com/events-and-presentations/",
            "text": "Conference call information",
            "relevance_score": 1.0,
        }]), None, 1)
        with patch("runtime.tool_registry._web_search", return_value=search_result) as web, patch(
            "runtime.tool_registry._web_sources", return_value=fetched
        ) as fetch, patch("runtime.tool_registry._academic_papers") as papers, patch(
            "runtime.tool_registry._academic_intelligence"
        ) as intelligence, patch("runtime.tool_registry._semantic_scholar") as semantic:
            results = _research_tools(queries, "DEEP_RESEARCH", False)

        self.assertEqual([result.name for result in results], [
            "research_source_plan", "web_search", "web_sources",
        ])
        web.assert_called_once()
        fetch.assert_called_once_with(search_result.output)
        papers.assert_not_called()
        intelligence.assert_not_called()
        semantic.assert_not_called()
        routed_queries = web.call_args.args[0]
        self.assertTrue(any("site:investor.nvidia.com" in query for query in routed_queries))
        self.assertTrue(any("site:nvidianews.nvidia.com" in query for query in routed_queries))
        self.assertTrue(any("site:sec.gov NVIDIA 8-K" in query for query in routed_queries))
        self.assertTrue(any("site:reuters.com NVIDIA earnings" in query for query in routed_queries))
        self.assertTrue(any("EPS revenue consensus guidance implied move" in query for query in routed_queries))

    def test_market_followup_searches_only_gap_queries_under_original_intent_gate(self) -> None:
        original = "NVIDIA 오늘 실적 발표와 시장 전망을 조사해줘"
        plan = research_source_plan(original)

        routed = _source_queries((original, "NVIDIA revenue consensus August 2026"), plan)

        self.assertEqual(routed, ("NVIDIA revenue consensus August 2026",))
        self.assertFalse(plan.academic_enabled)

    def test_academic_and_mixed_research_enable_academic_tools_only_when_requested(self) -> None:
        web = ToolResult("web_search", True, "[]", None, 1)
        sources = ToolResult("web_sources", True, "[]", None, 1)
        academic = ToolResult("academic_papers", True, "[]", None, 1)
        with patch("runtime.tool_registry._web_search", return_value=web), patch(
            "runtime.tool_registry._web_sources", return_value=sources
        ), patch("runtime.tool_registry._academic_papers", return_value=academic) as papers, patch(
            "runtime.tool_registry._academic_evidence_gaps", return_value=[]
        ):
            academic_results = _research_tools(
                ("AI가 주식시장에 미치는 영향에 대한 학술 논문을 조사해줘",), "DEEP_RESEARCH", False
            )
            mixed_results = _research_tools(
                ("NVIDIA 실적과 AI bubble에 대한 학계 연구를 함께 비교해줘",), "DEEP_RESEARCH", False
            )

        self.assertEqual(papers.call_count, 2)
        self.assertIn("academic_papers", [result.name for result in academic_results])
        self.assertIn("academic_papers", [result.name for result in mixed_results])
        mixed_plan = json.loads(mixed_results[0].output)
        self.assertIn("MIXED", mixed_plan["intents"])

    def test_current_market_relevance_excludes_academic_results_and_prioritizes_primary_sources(self) -> None:
        plan = research_source_plan("NVIDIA 오늘 실적 발표와 시장 전망을 조사해줘")
        ranked = _rank_relevant_web_results([
            {
                "provider": "brave", "title": "AI stock indices using 10-K filings",
                "url": "https://www.sciencedirect.com/science/article/example", "description": "NVIDIA AI paper",
            },
            {
                "provider": "brave", "title": "NVIDIA earnings call",
                "url": "https://investor.nvidia.com/events-and-presentations/", "description": "official event",
            },
            {
                "provider": "brave", "title": "NVIDIA earnings preview",
                "url": "https://www.reuters.com/technology/nvidia-preview", "description": "current consensus",
            },
        ], plan)

        self.assertEqual([item["relevance_score"] for item in ranked], [1.0, 0.95])
        self.assertFalse(any("sciencedirect" in str(item["url"]) for item in ranked))

    def test_current_market_relevance_limits_one_domain_from_crowding_out_other_tiers(self) -> None:
        plan = research_source_plan("NVIDIA 오늘 실적 발표와 시장 전망을 조사해줘")
        results = [
            {
                "provider": "brave", "title": f"SEC filing {index}",
                "url": f"https://www.sec.gov/Archives/filing-{index}.htm", "description": "earnings",
            }
            for index in range(5)
        ] + [{
            "provider": "brave", "title": "Reuters preview",
            "url": "https://www.reuters.com/business/nvidia-preview", "description": "consensus",
        }]
        ranked = _rank_relevant_web_results(results, plan)

        self.assertEqual(sum("sec.gov" in str(item["url"]) for item in ranked), 2)
        self.assertTrue(any("reuters.com" in str(item["url"]) for item in ranked))
    def setUp(self) -> None:
        self.fake_client = FakeClient()
        self.runtime = AgentRuntime(client=self.fake_client)
        self.temporary_directory = TemporaryDirectory()
        self.previous_user_store = web_app.user_store
        self.previous_project_store = web_app.project_store
        self.previous_attachment_directory = web_app.ATTACHMENT_DIRECTORY
        web_app.ATTACHMENT_DIRECTORY = Path(self.temporary_directory.name) / "uploads"
        project_data = Path(self.temporary_directory.name) / "project-data"
        (project_data / "projects").mkdir(parents=True)
        web_app.project_store = ProjectStore(
            Path(self.temporary_directory.name) / "projects.db",
            project_data,
            require_mount=False,
            client=self.fake_client,
        )
        web_app.user_store = UserStore(Path(self.temporary_directory.name) / "users.sqlite3")
        web_app.user_store.create("test-admin", "a-test-password", "admin")
        web_app.user_store.create("test-manager", "a-test-password", "manager")
        web_app.user_store.create("test-guest", "a-test-password", "guest")
        web_app.uploaded_attachments.clear()

    def tearDown(self) -> None:
        web_app.uploaded_attachments.clear()
        web_app.user_store = self.previous_user_store
        web_app.project_store = self.previous_project_store
        web_app.ATTACHMENT_DIRECTORY = self.previous_attachment_directory
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

    def test_runtime_sends_images_as_multimodal_content(self) -> None:
        result = self.runtime.chat(
            "이미지를 분석해줘.",
            "main",
            images=(("Uploaded image", "image/jpeg", b"image-bytes"),),
        )

        self.assertEqual(result.content, "verified response")
        user_content = self.fake_client.requests[-1]["json"]["messages"][-1]["content"]
        self.assertEqual(user_content[0], {"type": "text", "text": "이미지를 분석해줘."})
        self.assertEqual(user_content[1], {"type": "text", "text": "Uploaded image"})
        self.assertTrue(user_content[2]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

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

    def test_role_specific_session_lifetimes(self) -> None:
        admin = TestClient(web_app.app)
        manager = TestClient(web_app.app)
        guest = TestClient(web_app.app)
        admin_login = admin.post("/api/login", json={"username": "test-admin", "password": "a-test-password"})
        manager_login = manager.post("/api/login", json={"username": "test-manager", "password": "a-test-password"})
        guest_login = guest.post("/api/login", json={"username": "test-guest", "password": "a-test-password"})

        self.assertIn("Max-Age=86400", admin_login.headers["set-cookie"])
        self.assertIn("Max-Age=1800", manager_login.headers["set-cookie"])
        self.assertIn("Max-Age=900", guest_login.headers["set-cookie"])
        self.assertEqual(admin.get("/api/me").json()["session_idle_timeout_seconds"], 86_400)
        self.assertEqual(manager.get("/api/me").json()["session_idle_timeout_seconds"], 1_800)
        self.assertEqual(guest.get("/api/me").json()["session_idle_timeout_seconds"], 900)
        self.assertEqual(admin.get("/api/agents").json(), manager.get("/api/agents").json())

    def test_existing_user_database_is_migrated_for_manager_role(self) -> None:
        database_path = Path(self.temporary_directory.name) / "legacy-users.sqlite3"
        with sqlite3.connect(database_path) as connection:
            connection.execute(
                """CREATE TABLE users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('admin', 'guest')),
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL
                )"""
            )
            connection.execute(
                "INSERT INTO users VALUES (?, ?, ?, ?, ?)",
                ("legacy-admin", "stored-hash", "admin", 1, "2026-01-01T00:00:00+00:00"),
            )

        store = UserStore(database_path)
        manager = store.create("new-manager", "a-test-password", "manager")
        with sqlite3.connect(database_path) as connection:
            usernames = {row[0] for row in connection.execute("SELECT username FROM users")}

        self.assertEqual(manager.role, "manager")
        self.assertEqual(usernames, {"legacy-admin", "new-manager"})

    def test_guest_cannot_upload_files(self) -> None:
        guest = self.authenticated_client("test-guest")

        response = guest.post("/api/upload", files={"file": ("notes.txt", b"private document fact")})

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"], "guest accounts cannot upload files")

    def test_uploaded_text_is_bound_to_browser_and_consumed_by_chat(self) -> None:
        previous_runtime = web_app.runtime
        web_app.runtime = self.runtime
        try:
            owner = self.authenticated_client()
            other_browser = self.authenticated_client()
            upload = owner.post("/api/upload", files={"file": ("notes.txt", "private document fact", "text/plain")})
            attachment_id = upload.json()["attachment_id"]
            forbidden = other_browser.post("/api/chat", json={"message": "summarize", "attachment_ids": [attachment_id]})
            response = owner.post("/api/chat", json={"message": "summarize", "selected_agent": "main", "attachment_ids": [attachment_id]})
            expired = owner.post("/api/chat", json={"message": "again", "selected_agent": "main", "attachment_ids": [attachment_id]})
        finally:
            web_app.runtime = previous_runtime

        self.assertEqual(upload.status_code, 200)
        self.assertEqual(forbidden.status_code, 403)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(expired.status_code, 404)
        model_messages = self.fake_client.requests[-1]["json"]["messages"]
        self.assertIn("private document fact", model_messages[-1]["content"])
        self.assertIn("분석 대상 데이터로만 취급", model_messages[-1]["content"])

    def test_uploaded_attachment_survives_web_process_restart(self) -> None:
        previous_runtime = web_app.runtime
        web_app.runtime = self.runtime
        try:
            client = self.authenticated_client()
            upload = client.post("/api/upload", files={"file": ("notes.txt", b"survives restart")})
            attachment_id = upload.json()["attachment_id"]
            web_app.uploaded_attachments.clear()
            response = client.post("/api/chat", json={
                "message": "summarize",
                "selected_agent": "main",
                "attachment_ids": [attachment_id],
            })
        finally:
            web_app.runtime = previous_runtime

        self.assertEqual(response.status_code, 200)
        self.assertIn("survives restart", self.fake_client.requests[-1]["json"]["messages"][-1]["content"])
        self.assertFalse((web_app.ATTACHMENT_DIRECTORY / f"{attachment_id}.json").exists())

    def test_generated_image_continuation_is_retained_for_twenty_four_hours(self) -> None:
        attachment_id = "generated-continuation-id"
        web_app.save_attachment(
            attachment_id,
            web_app.UploadedAttachment(
                "owner",
                "pose-corrected.png",
                web_app.GENERATED_IMAGE_TEXT,
                False,
                (("Original image", "image/png", b"source"),),
                web_app.time() - web_app.ATTACHMENT_TTL_SECONDS - 1,
            ),
        )

        web_app.prune_attachments()

        self.assertIsNotNone(web_app.load_attachment(attachment_id))

    def test_uploaded_python_can_auto_route_to_coding_without_local_tools(self) -> None:
        previous_runtime = web_app.runtime
        web_app.runtime = self.runtime
        try:
            client = self.authenticated_client()
            upload = client.post("/api/upload", files={"file": ("scanner.py", b"def scan(): return 'ok'")})
            attachment_id = upload.json()["attachment_id"]
            with patch("runtime.agent_runtime.run_agent_tools", return_value=[]) as run_tools:
                response = client.post("/api/chat", json={
                    "message": "이 코드가 작동하는지 검토해줘.",
                    "selected_agent": "auto",
                    "attachment_ids": [attachment_id],
                })
        finally:
            web_app.runtime = previous_runtime

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["activity"]["routed_agent"], "coding")
        run_tools.assert_called_once()
        self.assertEqual(run_tools.call_args.args[0], "coding")
        self.assertFalse(run_tools.call_args.args[3])

    def test_project_api_persists_conversation_memory_and_owner_scope(self) -> None:
        previous_runtime = web_app.runtime
        web_app.runtime = self.runtime
        try:
            owner = self.authenticated_client()
            created = owner.post("/api/projects", json={"name": "Persistent Test", "description": "Research"})
            self.assertEqual(created.status_code, 200)
            project_id = created.json()["id"]
            conversation = owner.post(
                f"/api/projects/{project_id}/conversations", json={"title": "Pressure decision"}
            )
            conversation_id = conversation.json()["id"]
            response = owner.post("/api/chat", json={
                "message": "테스트 압력은 12 bar로 결정하자.",
                "selected_agent": "main",
                "project_id": project_id,
                "conversation_id": conversation_id,
            })
            memory = owner.post(f"/api/projects/{project_id}/memories", json={
                "type": "decision", "content": "Test pressure is 12 bar", "confidence": "HIGH",
            })
            messages = owner.get(f"/api/projects/{project_id}/conversations/{conversation_id}/messages")
            other = self.authenticated_client("test-manager")
            forbidden = other.get(f"/api/projects/{project_id}")
        finally:
            web_app.runtime = previous_runtime

        self.assertEqual(response.status_code, 200)
        self.assertEqual([item["role"] for item in messages.json()["messages"]], ["user", "assistant"])
        self.assertEqual(memory.status_code, 200)
        self.assertEqual(forbidden.status_code, 404)

    def test_project_api_allows_only_owner_to_delete_project(self) -> None:
        owner = self.authenticated_client()
        project_id = owner.post("/api/projects", json={"name": "Disposable API Project"}).json()["id"]
        other = self.authenticated_client("test-manager")

        forbidden = other.delete(f"/api/projects/{project_id}")
        deleted = owner.delete(f"/api/projects/{project_id}")
        missing = owner.get(f"/api/projects/{project_id}")

        self.assertEqual(forbidden.status_code, 404)
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(deleted.json(), {"status": "deleted"})
        self.assertEqual(missing.status_code, 404)

    def test_project_upload_and_image_artifact_persist_to_project_files(self) -> None:
        client = self.authenticated_client()
        project_id = client.post("/api/projects", json={"name": "Artifacts"}).json()["id"]
        conversation_id = client.post(
            f"/api/projects/{project_id}/conversations", json={"title": "Files"}
        ).json()["id"]
        upload = client.post(
            "/api/upload",
            data={"project_id": project_id, "conversation_id": conversation_id},
            files={"file": ("notes.txt", b"critical pressure is 12 bar", "text/plain")},
        )
        generated = GeneratedImage(b"png-data", 42, "generate", "generate-42.png")
        with patch("web.app.build_image_prompt", side_effect=planned_prompt), patch(
            "web.app.assess_image_quality", return_value=PASSED_IMAGE_QUALITY
        ), patch("web.app.create_image", return_value=generated):
            image = client.post("/api/chat", json={
                "message": "/image test diagram",
                "project_id": project_id,
                "conversation_id": conversation_id,
            })
        files = client.get(f"/api/projects/{project_id}/files").json()["files"]
        downloaded = client.get(f"/api/projects/{project_id}/files/{upload.json()['project_file_id']}")

        self.assertEqual(upload.status_code, 200)
        self.assertEqual(image.status_code, 200)
        self.assertEqual(downloaded.content, b"critical pressure is 12 bar")
        self.assertEqual({item["original_name"] for item in files}, {"notes.txt", "generate-42.png"})
        self.assertTrue(next(item for item in files if item["original_name"] == "generate-42.png")["artifact_id"])

    def test_oversized_project_upload_is_rejected_before_persistence(self) -> None:
        client = self.authenticated_client()
        project_id = client.post("/api/projects", json={"name": "Bounded upload"}).json()["id"]
        conversation_id = client.post(
            f"/api/projects/{project_id}/conversations", json={"title": "Files"}
        ).json()["id"]

        upload = client.post(
            "/api/upload",
            data={"project_id": project_id, "conversation_id": conversation_id},
            files={"file": ("too-large.txt", b"x" * (10 * 1024 * 1024 + 1), "text/plain")},
        )
        files = client.get(f"/api/projects/{project_id}/files").json()["files"]

        self.assertEqual(upload.status_code, 422)
        self.assertEqual(files, [])

    def test_attachment_quota_rejects_project_upload_before_persistence(self) -> None:
        client = self.authenticated_client()
        project_id = client.post("/api/projects", json={"name": "Quota ordering"}).json()["id"]
        conversation_id = client.post(
            f"/api/projects/{project_id}/conversations", json={"title": "Files"}
        ).json()["id"]
        for index in range(web_app.MAX_ATTACHMENTS_PER_CHAT):
            response = client.post(
                "/api/upload", files={"file": (f"pending-{index}.txt", b"pending", "text/plain")}
            )
            self.assertEqual(response.status_code, 200)

        upload = client.post(
            "/api/upload",
            data={"project_id": project_id, "conversation_id": conversation_id},
            files={"file": ("blocked.txt", b"must not persist", "text/plain")},
        )
        files = client.get(f"/api/projects/{project_id}/files").json()["files"]

        self.assertEqual(upload.status_code, 429)
        self.assertEqual(files, [])

    def test_project_storage_offline_is_explicit_and_general_chat_is_unaffected(self) -> None:
        previous_store = web_app.project_store
        web_app.project_store = ProjectStore(
            Path(self.temporary_directory.name) / "offline.db",
            Path(self.temporary_directory.name) / "missing-project-storage",
            require_mount=True,
        )
        try:
            client = self.authenticated_client()
            storage = client.get("/api/projects/storage")
            project = client.post("/api/projects", json={"name": "Blocked"})
            with patch("web.app.runtime.chat", return_value=self.runtime.chat("hello", "main")):
                general = client.post("/api/chat", json={"message": "hello", "selected_agent": "main"})
        finally:
            web_app.project_store = previous_store

        self.assertFalse(storage.json()["online"])
        self.assertEqual(project.status_code, 503)
        self.assertEqual(general.status_code, 200)

    def test_web_image_command_returns_renderable_png(self) -> None:
        client = self.authenticated_client()
        generated = GeneratedImage(b"png", 42, "generate", "generate-42.png")

        with patch("web.app.build_image_prompt", side_effect=planned_prompt), patch(
            "web.app.assess_image_quality", return_value=PASSED_IMAGE_QUALITY
        ), patch("web.app.create_image", return_value=generated) as create:
            response = client.post("/api/chat", json={"message": "/image glass tower"})

        self.assertEqual(response.status_code, 200)
        create.assert_called_once_with("glass tower", None)
        payload = response.json()
        self.assertEqual(payload["activity"]["routed_agent"], "image")
        self.assertTrue(payload["generated_images"][0]["data_url"].startswith("data:image/png;base64,"))

    def test_web_edit_command_uses_uploaded_image(self) -> None:
        client = self.authenticated_client()
        extracted = ExtractedUpload(
            "OCR text", False, (("Uploaded image", "image/jpeg", b"normalized-source"),)
        )
        with patch("web.app.extract_text", return_value=extracted), patch(
            "web.app.image_thumbnail_data_url", return_value="data:image/jpeg;base64,dGh1bWI="
        ):
            upload = client.post("/api/upload", files={"file": ("source.png", b"source", "image/png")})
        attachment_id = upload.json()["attachment_id"]
        generated = GeneratedImage(b"edited", 7, "edit", "edit-7.png")
        plan = ImageEditPlan((ImageEditOperation(
            "generic_edit", "image", "Apply the requested nighttime change.", "generative_edit"
        ),))
        completed = ImageEditCompletion(True, True, (("generic_edit", True),), True, "complete")

        with patch("runtime.image_client.infer_image_intent", return_value="edit"), patch(
            "web.app.build_image_edit_plan", return_value=plan
        ), patch(
            "web.app.build_image_prompt", side_effect=planned_prompt
        ), patch("web.app.assess_image_edit_completion", return_value=completed), patch(
            "web.app.create_image", return_value=generated
        ) as create:
            response = client.post("/api/chat", json={
                "message": "/edit make it nighttime",
                "attachment_ids": [attachment_id],
            })

        self.assertEqual(response.status_code, 200)
        create.assert_called_once_with(unittest.mock.ANY, b"normalized-source")
        self.assertEqual(response.json()["activity"]["route_summary"], "Remote image edit")
        self.assertEqual(upload.json()["thumbnail_data_url"], "data:image/jpeg;base64,dGh1bWI=")

    def test_natural_edit_request_with_image_routes_to_local_image_service(self) -> None:
        client = self.authenticated_client()
        extracted = ExtractedUpload(
            "OCR text", False, (("Uploaded image", "image/jpeg", b"normalized-source"),)
        )
        with patch("web.app.extract_text", return_value=extracted), patch(
            "web.app.image_thumbnail_data_url", return_value="data:image/jpeg;base64,dGh1bWI="
        ):
            upload = client.post("/api/upload", files={"file": ("portrait.png", b"source", "image/png")})
        attachment_id = upload.json()["attachment_id"]
        prompt = "배경을 자연스럽게 바꿔주고 조명을 보정해줘."
        generated = GeneratedImage(b"edited", 8, "edit", "edit-8.png")
        plan = ImageEditPlan((
            ImageEditOperation("background_edit", "background", "Change the background.", "generative_edit"),
            ImageEditOperation("lighting_edit", "lighting", "Correct the lighting.", "generative_edit"),
        ))
        completed = ImageEditCompletion(
            True, True, (("background_edit", True), ("lighting_edit", True)), True, "complete"
        )

        with patch("web.app.build_image_edit_plan", return_value=plan), patch(
            "web.app.build_image_prompt", side_effect=planned_prompt
        ), patch(
            "web.app.assess_image_edit_completion", return_value=completed
        ), patch("web.app.create_image", return_value=generated) as create:
            response = client.post("/api/chat", json={
                "message": prompt,
                "selected_agent": "research",
                "attachment_ids": [attachment_id],
            })

        self.assertEqual(response.status_code, 200)
        create.assert_called_once_with(unittest.mock.ANY, b"normalized-source")
        self.assertEqual(response.json()["activity"]["routed_agent"], "image")
        continuation_image_id = response.json()["continuation_image_id"]

        with patch("runtime.image_client.infer_image_intent", return_value="resend"):
            resend = client.post("/api/chat", json={
                "message": "사진 다시 보내줘야지",
                "selected_agent": "main",
                "session_id": response.json()["session_id"],
                "continuation_image_id": continuation_image_id,
            })
        self.assertEqual(resend.status_code, 200)
        self.assertEqual(resend.json()["activity"]["route_summary"], "Remote image resend")
        self.assertEqual(resend.json()["continuation_image_id"], continuation_image_id)

        correction = "고개를 똑바로 보게 수정해달라니까. 과도하게 바뀐 외모도 원래대로 되돌려줘."
        revised = GeneratedImage(b"revised", 0, "pose", "pose-corrected.png")
        pose_plan = ImageEditPlan((ImageEditOperation(
            "face_orientation", "face_orientation", "Turn the face toward the camera.", "pose_correction"
        ),))
        pose_completed = ImageEditCompletion(True, True, (("face_orientation", True),), True, "frontal")
        with patch("runtime.image_client.infer_image_intent", return_value="pose"), patch(
            "web.app.build_image_edit_plan", return_value=pose_plan
        ), patch(
            "web.app.build_image_prompt", side_effect=planned_prompt
        ), patch(
            "web.app.assess_image_edit_completion", return_value=pose_completed
        ), patch(
            "web.app.correct_portrait_pose", return_value=revised
        ) as correct, patch(
            "web.app.create_image"
        ) as diffuse:
            follow_up = client.post("/api/chat", json={
                "message": correction,
                "selected_agent": "main",
                "session_id": response.json()["session_id"],
                "continuation_image_id": continuation_image_id,
            })

        self.assertEqual(follow_up.status_code, 200)
        correct.assert_called_once_with(b"normalized-source")
        diffuse.assert_not_called()
        self.assertNotEqual(follow_up.json()["continuation_image_id"], continuation_image_id)
        self.assertEqual(follow_up.json()["activity"]["route_summary"], "Remote portrait pose correction")

    def test_severe_generation_feedback_regenerates_without_failed_image_anchor(self) -> None:
        client = self.authenticated_client()
        initial = GeneratedImage(b"failed-image", 11, "generate", "generate-11.png")
        regenerated = GeneratedImage(b"better-image", 12, "generate", "generate-12.png")
        request = "예쁜 여학생이 스케이트보드를 타는 그림을 그려줘. 일본 애니메이션 스타일로."

        with patch("web.app.build_image_prompt", side_effect=planned_prompt), patch(
            "web.app.assess_image_quality", return_value=PASSED_IMAGE_QUALITY
        ), patch("web.app.create_image", side_effect=[initial, regenerated]) as create:
            first = client.post("/api/chat", json={"message": f"/image {request}"})
            with patch("runtime.image_client.infer_image_intent", return_value="regenerate"):
                second = client.post("/api/chat", json={
                    "message": "얼굴이 망가졌네. 그림을 제대로 다시 그려줘.",
                    "session_id": first.json()["session_id"],
                    "continuation_image_id": first.json()["continuation_image_id"],
                })

        self.assertEqual(second.status_code, 200)
        self.assertIsNone(create.call_args_list[1].args[1])
        self.assertEqual(second.json()["activity"]["image"]["mode"], "regenerate")
        self.assertIn("severe quality failure", second.json()["activity"]["image"]["reason"])

    def test_quality_gate_allows_only_one_source_free_regenerate(self) -> None:
        client = self.authenticated_client()
        failed = ImageQualityAssessment(True, False, ("face", "anatomy"), "warped face and limbs")
        generated = [
            GeneratedImage(b"first", 21, "generate", "generate-21.png"),
            GeneratedImage(b"retry", 22, "generate", "generate-22.png"),
        ]

        with patch("web.app.build_image_prompt", side_effect=planned_prompt) as planner, patch(
            "web.app.assess_image_quality", side_effect=[failed, PASSED_IMAGE_QUALITY]
        ), patch("web.app.create_image", side_effect=generated) as create:
            response = client.post("/api/chat", json={"message": "/image anime full-body action"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(create.call_count, 2)
        self.assertIsNone(create.call_args_list[1].args[1])
        self.assertTrue(planner.call_args_list[1].kwargs["simplify_composition"])
        image_activity = response.json()["activity"]["image"]
        self.assertEqual(image_activity["mode"], "regenerate")
        self.assertEqual(image_activity["retry_policy"], "internal regenerate")

    def test_quality_sensitive_generation_selects_best_of_two_candidates(self) -> None:
        client = self.authenticated_client()
        plan = ImagePromptPlan(
            "polished portrait", "person", "walking", "anime", "high", "high", "none",
            quality_sensitive=True,
        )
        weaker = ImageQualityAssessment(
            True, True, (), "acceptable", (("face", 6), ("overall_appeal", 6)), 6.0, "accept"
        )
        stronger = ImageQualityAssessment(
            True, True, (), "strong", (("face", 9), ("overall_appeal", 9)), 9.0, "accept"
        )
        candidates = [
            GeneratedImage(b"weaker", 31, "generate", "generate-31.png"),
            GeneratedImage(b"stronger", 32, "generate", "generate-32.png"),
        ]

        with patch("web.app.build_image_prompt", return_value=plan), patch(
            "web.app.assess_image_quality", side_effect=[weaker, stronger]
        ), patch("web.app.create_image", side_effect=candidates) as create:
            response = client.post("/api/chat", json={"message": "/image attractive anime character walking"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(create.call_count, 2)
        self.assertEqual(response.json()["generated_images"][0]["filename"], "generate-32.png")
        activity = response.json()["activity"]["image"]
        self.assertEqual(activity["mode"], "generate")
        self.assertEqual(activity["retry_policy"], "candidate selection")
        self.assertEqual([candidate["score"] for candidate in activity["candidates"]], [6.0, 9.0])

    def test_project_preferences_and_explicit_reference_research_feed_scene_planner(self) -> None:
        client = self.authenticated_client()
        project_id = client.post("/api/projects", json={"name": "Visual preferences"}).json()["id"]
        conversation_id = client.post(
            f"/api/projects/{project_id}/conversations", json={"title": "Image"}
        ).json()["id"]
        client.post(f"/api/projects/{project_id}/memories", json={
            "type": "preference", "content": "Prefers clean anime faces and readable full-body action",
        })
        generated = GeneratedImage(b"image", 41, "generate", "generate-41.png")
        references = [{
            "provider": "brave", "title": "Pose reference", "url": "https://example.com/reference",
            "description": "Clear three-quarter walking pose and restrained background",
        }]

        with patch("web.app.visual_search", side_effect=RuntimeError("visual unavailable")), patch(
            "web.app.search", return_value=references
        ) as search_reference, patch(
            "web.app.build_image_prompt", side_effect=planned_prompt
        ) as planner, patch("web.app.assess_image_quality", return_value=PASSED_IMAGE_QUALITY), patch(
            "web.app.create_image", return_value=generated
        ):
            response = client.post("/api/chat", json={
                "message": "/image 검색해서 참고한 애니메이션 인물 전신을 그려줘",
                "project_id": project_id,
                "conversation_id": conversation_id,
            })

        self.assertEqual(response.status_code, 200)
        search_reference.assert_called_once()
        self.assertIn("clean anime faces", planner.call_args.kwargs["preference_context"])
        self.assertIn("Clear three-quarter walking pose", planner.call_args.kwargs["reference_cues"])
        activity = response.json()["activity"]["image"]
        self.assertTrue(activity["preferences_applied"])
        self.assertEqual(activity["reference_research"]["source_count"], 1)

    def test_explicit_visual_preference_is_saved_after_project_image_generation(self) -> None:
        client = self.authenticated_client()
        project_id = client.post("/api/projects", json={"name": "Image memory"}).json()["id"]
        conversation_id = client.post(
            f"/api/projects/{project_id}/conversations", json={"title": "Preferences"}
        ).json()["id"]
        generated = GeneratedImage(b"image", 51, "generate", "generate-51.png")
        preference = "앞으로 그림은 항상 깔끔한 애니 스타일과 자연스러운 인체를 선호해"

        with patch("web.app.build_image_prompt", side_effect=planned_prompt), patch(
            "web.app.assess_image_quality", return_value=PASSED_IMAGE_QUALITY
        ), patch("web.app.create_image", return_value=generated):
            response = client.post("/api/chat", json={
                "message": f"/image {preference}",
                "project_id": project_id,
                "conversation_id": conversation_id,
            })

        memories = client.get(f"/api/projects/{project_id}/memories").json()["memories"]
        self.assertEqual(response.status_code, 200)
        self.assertTrue(any(memory["type"] == "preference" and memory["content"] == f"/image {preference}" for memory in memories))

    def test_front_facing_feedback_uses_identity_preserving_pose_service(self) -> None:
        client = self.authenticated_client()
        extracted = ExtractedUpload(
            "OCR text", False, (("Uploaded image", "image/jpeg", b"original-face"),)
        )
        with patch("web.app.extract_text", return_value=extracted), patch(
            "web.app.image_thumbnail_data_url", return_value="data:image/jpeg;base64,dGh1bWI="
        ):
            upload = client.post("/api/upload", files={"file": ("portrait.png", b"source", "image/png")})

        corrected = GeneratedImage(b"frontal", 0, "pose", "pose-corrected.png")
        plan = ImageEditPlan((ImageEditOperation(
            "face_orientation", "face_orientation", "Turn the face toward the camera.", "pose_correction"
        ),))
        completed = ImageEditCompletion(True, True, (("face_orientation", True),), True, "frontal")
        with patch("runtime.image_client.infer_image_intent", return_value="pose"), patch(
            "web.app.build_image_edit_plan", return_value=plan
        ), patch(
            "web.app.build_image_prompt", side_effect=planned_prompt
        ), patch(
            "web.app.assess_image_edit_completion", return_value=completed
        ), patch(
            "web.app.correct_portrait_pose", return_value=corrected
        ) as correct, patch(
            "web.app.create_image"
        ) as diffuse:
            response = client.post("/api/chat", json={
                "message": "얼굴을 증명사진처럼 정면을 쳐다보게 바꿔줘.",
                "selected_agent": "main",
                "attachment_ids": [upload.json()["attachment_id"]],
            })

        self.assertEqual(response.status_code, 200)
        correct.assert_called_once_with(b"original-face")
        diffuse.assert_not_called()
        self.assertEqual(response.json()["activity"]["route_summary"], "Remote portrait pose correction")
        self.assertEqual(response.json()["content"], "요청한 이미지 수정 사항을 모두 적용했습니다.")

    def test_multi_intent_edit_executes_pose_then_appearance_and_checks_both(self) -> None:
        client = self.authenticated_client()
        extracted = ExtractedUpload(
            "OCR text", False, (("Uploaded image", "image/jpeg", b"original-face"),)
        )
        with patch("web.app.extract_text", return_value=extracted), patch(
            "web.app.image_thumbnail_data_url", return_value="data:image/jpeg;base64,dGh1bWI="
        ):
            upload = client.post("/api/upload", files={"file": ("portrait.png", b"source", "image/png")})
        plan = ImageEditPlan((
            ImageEditOperation(
                "face_orientation", "face_orientation", "Turn the face toward the camera.", "pose_correction"
            ),
            ImageEditOperation(
                "appearance_refinement", "face", "Naturally refine facial attractiveness.", "generative_edit"
            ),
        ))
        frontal = GeneratedImage(b"frontal-intermediate", 0, "pose", "pose-corrected.png")
        final = GeneratedImage(b"handsome-frontal", 81, "edit", "edit-81.png")
        completed = ImageEditCompletion(
            True, True, (("face_orientation", True), ("appearance_refinement", True)), True, "all complete"
        )
        with patch("runtime.image_client.infer_image_intent", return_value="edit"), patch(
            "web.app.build_image_edit_plan", return_value=plan
        ), patch("web.app.build_image_prompt", side_effect=planned_prompt), patch(
            "web.app.correct_portrait_pose", return_value=frontal
        ) as pose, patch("web.app.create_image", return_value=final) as edit, patch(
            "web.app.assess_image_edit_completion", return_value=completed
        ) as completion:
            response = client.post("/api/chat", json={
                "message": "이 남자의 얼굴을 좀 더 잘생기게 하고 얼굴을 정면으로 바라보게 해줘.",
                "attachment_ids": [upload.json()["attachment_id"]],
            })

        self.assertEqual(response.status_code, 200)
        pose.assert_called_once_with(b"original-face")
        edit.assert_called_once_with(unittest.mock.ANY, b"frontal-intermediate", 0.25)
        completion.assert_called_once_with(plan, b"original-face", b"handsome-frontal")
        activity = response.json()["activity"]["image"]["edit_plan"]
        self.assertEqual(activity["tools"], ["portrait.frontalize", "image.edit"])
        self.assertEqual(activity["status"], {"face_orientation": True, "appearance_refinement": True})
        self.assertTrue(activity["identity_preserved"])

    def test_multi_intent_edit_retries_incomplete_modifications_once(self) -> None:
        client = self.authenticated_client()
        extracted = ExtractedUpload(
            "OCR text", False, (("Uploaded image", "image/jpeg", b"original-face"),)
        )
        with patch("web.app.extract_text", return_value=extracted), patch(
            "web.app.image_thumbnail_data_url", return_value="data:image/jpeg;base64,dGh1bWI="
        ):
            upload = client.post("/api/upload", files={"file": ("portrait.png", b"source", "image/png")})
        plan = ImageEditPlan((
            ImageEditOperation("hair_edit", "hair", "Make the hair longer.", "generative_edit"),
            ImageEditOperation("clothing_edit", "clothing", "Make the clothing black.", "generative_edit"),
        ))
        first = GeneratedImage(b"hair-only", 91, "edit", "edit-91.png")
        retry = GeneratedImage(b"hair-and-clothing", 92, "edit", "edit-92.png")
        incomplete = ImageEditCompletion(
            True, False, (("hair_edit", True), ("clothing_edit", False)), True, "clothing unchanged"
        )
        complete = ImageEditCompletion(
            True, True, (("hair_edit", True), ("clothing_edit", True)), True, "all complete"
        )
        with patch("runtime.image_client.infer_image_intent", return_value="edit"), patch(
            "web.app.build_image_edit_plan", return_value=plan
        ), patch("web.app.build_image_prompt", side_effect=planned_prompt) as planner, patch(
            "web.app.create_image", side_effect=[first, retry]
        ) as edit, patch(
            "web.app.assess_image_edit_completion", side_effect=[incomplete, complete]
        ) as completion:
            response = client.post("/api/chat", json={
                "message": "머리를 길게 만들고 옷을 검정색으로 바꿔줘",
                "attachment_ids": [upload.json()["attachment_id"]],
            })

        self.assertEqual(response.status_code, 200)
        self.assertEqual(edit.call_count, 2)
        self.assertEqual(edit.call_args_list[1].args[1], b"hair-only")
        self.assertIn("clothing_edit", planner.call_args_list[1].args[0])
        self.assertEqual(completion.call_count, 2)
        activity = response.json()["activity"]["image"]["edit_plan"]
        self.assertEqual(activity["tools"], ["image.edit", "image.edit.retry"])
        self.assertTrue(activity["passed"])

    def test_upload_rejects_unsupported_and_oversized_files(self) -> None:
        client = self.authenticated_client()

        unsupported = client.post("/api/upload", files={"file": ("payload.exe", b"not allowed")})
        oversized = client.post("/api/upload", files={"file": ("large.txt", b"x" * (10 * 1024 * 1024 + 1))})

        self.assertEqual(unsupported.status_code, 422)
        self.assertIn("unsupported file type", unsupported.json()["detail"])
        self.assertEqual(oversized.status_code, 422)
        self.assertIn("10 MB", oversized.json()["detail"])

    def test_web_ui_assets_are_not_cached(self) -> None:
        client = TestClient(web_app.app)

        index = client.get("/", follow_redirects=False)
        login = client.get("/login")
        script = client.get("/static/app.js")
        answer_styles = client.get("/static/answer-rendering.css")

        self.assertEqual(index.status_code, 303)
        self.assertEqual(login.headers["cache-control"], "no-store")
        self.assertEqual(script.headers["cache-control"], "no-store")
        self.assertEqual(answer_styles.status_code, 200)
        self.assertEqual(answer_styles.headers["cache-control"], "no-store")

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

    def test_admin_browser_blocks_direct_server_and_falls_back_for_auto_server_route(self) -> None:
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
        self.assertEqual(automatic.status_code, 200)
        self.assertEqual(automatic.json()["activity"]["routed_agent"], "main")
        run_tools.assert_not_called()

    def test_guest_auto_server_route_falls_back_without_server_tools(self) -> None:
        previous_runtime = web_app.runtime
        web_app.runtime = self.runtime
        try:
            guest = self.authenticated_client("test-guest")
            with patch("runtime.agent_runtime.run_agent_tools") as run_tools:
                response = guest.post("/api/chat", json={"message": "현재 GPU 상태를 알려줘"})
        finally:
            web_app.runtime = previous_runtime

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["activity"]["routed_agent"], "main")
        run_tools.assert_not_called()

    def test_admin_auto_coding_route_falls_back_to_main(self) -> None:
        previous_runtime = web_app.runtime
        web_app.runtime = self.runtime
        try:
            admin = self.authenticated_client()
            with patch("runtime.agent_runtime.run_agent_tools") as run_tools:
                response = admin.post("/api/chat", json={"message": "이 코드 파일의 오류를 설명해줘"})
        finally:
            web_app.runtime = previous_runtime

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["activity"]["routed_agent"], "main")
        run_tools.assert_not_called()

    def test_guest_research_has_no_local_project_tools(self) -> None:
        previous_runtime = web_app.runtime
        web_app.runtime = self.runtime
        try:
            guest = self.authenticated_client("test-guest")
            with patch("runtime.agent_runtime.execute_research_action", return_value=[]) as run_tools:
                response = guest.post("/api/chat", json={"message": "수소 연구를 요약해줘", "selected_agent": "research"})
        finally:
            web_app.runtime = previous_runtime

        self.assertEqual(response.status_code, 200)
        self.assertEqual(run_tools.call_count, 1)
        self.assertEqual(run_tools.call_args.args[:3], ("SEARCH_WEB", ("수소 연구를 요약해줘",), "searxng"))
        self.assertEqual(run_agent_tools("research", "수소 연구를 요약해줘", allow_local_tools=False), [])

    def test_admin_browser_research_has_no_local_project_tools(self) -> None:
        previous_runtime = web_app.runtime
        web_app.runtime = self.runtime
        try:
            admin = self.authenticated_client()
            with patch("runtime.agent_runtime.execute_research_action", return_value=[]) as run_tools:
                response = admin.post("/api/chat", json={"message": "수소 연구를 요약해줘", "selected_agent": "research"})
        finally:
            web_app.runtime = previous_runtime

        self.assertEqual(response.status_code, 200)
        self.assertEqual(run_tools.call_count, 1)
        self.assertEqual(run_tools.call_args.args[:3], ("SEARCH_WEB", ("수소 연구를 요약해줘",), "searxng"))

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
        with patch.dict(os.environ, {
            "SEARXNG_URL": "", "SERPER_API_KEY": "", "BRAVE_SEARCH_API_KEY": "",
        }):
            result = runtime.chat("오늘 서울 날씨는 어때?", "auto")

        self.assertEqual(result.route.agent, "research")
        self.assertEqual(result.route.search_mode, "QUICK_SEARCH")
        self.assertEqual(result.tools[-1]["name"], "web_search")
        self.assertFalse(result.tools[-1]["success"])
        self.assertIn("search provider unavailable: searxng", result.tools[-1]["error"])
        self.assertEqual(result.research["rounds"][0]["provider"], "searxng")

    def test_model_search_decision_controls_search_mode(self) -> None:
        client = SearchDecisionClient()
        runtime = AgentRuntime(client=client)

        self.assertEqual(runtime._search_mode("현재 repository 구조를 실제로 확인해줘"), "QUICK_SEARCH")
        decision_request = client.requests[0]["json"]
        self.assertEqual(decision_request["max_tokens"], 256)
        planner_prompt = decision_request["messages"][0]["content"]
        self.assertIn("supply-chain or value-chain relationship", planner_prompt)
        self.assertIn("pricing/volume/mix/margin transmission", planner_prompt)
        self.assertIn("evidence supporting each causal premise", planner_prompt)
        self.assertEqual(
            decision_request["messages"][1],
            {"role": "user", "content": "현재 repository 구조를 실제로 확인해줘"},
        )

    def test_search_mode_prompt_requires_external_premises_for_real_world_impact_analysis(self) -> None:
        client = SearchDecisionClient()
        runtime = AgentRuntime(client=client)

        runtime._search_decision("A사의 실적이 B산업에 미칠 영향을 분석해줘")

        planner_prompt = client.requests[0]["json"]["messages"][0]["content"]
        self.assertIn("materially depend on external facts", planner_prompt)
        self.assertIn("impact analysis", planner_prompt)
        self.assertIn("Do not confuse permission to reason", planner_prompt)
        self.assertIn("Current UTC date", planner_prompt)

    def test_model_deep_research_decision_runs_search_with_model_query(self) -> None:
        class DeepResearchDecisionClient(FakeClient):
            def post(self, url: str, json: dict[str, object]) -> FakeResponse:
                self.requests.append({"url": url, "json": json})
                responses = (
                    '{"search_mode":"DEEP_RESEARCH","queries":["liquefied hydrogen storage papers 2024 2026","liquefied hydrogen storage review"],"focus":["recent papers","reviews"]}',
                    '{"next_action":"SEARCH_ACADEMIC","queries":["liquefied hydrogen storage papers 2024 2026"],'
                    '"provider":"","unresolved_questions":["recent papers"],"decision_summary":"Find scholarly evidence",'
                    '"ready_to_answer":false,"complexity":"MODERATE","use_critic":false}',
                    '{"next_action":"FINAL_ANSWER","queries":[],"provider":"","unresolved_questions":[],'
                    '"decision_summary":"Evidence sufficient","ready_to_answer":true,"complexity":"MODERATE","use_critic":false}',
                    "web-verified answer",
                )
                content = responses[len(self.requests) - 1]
                response = FakeResponse()
                response.json = lambda: {  # type: ignore[method-assign]
                    "choices": [{"message": {"content": content}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4},
                }
                return response

        runtime = AgentRuntime(client=DeepResearchDecisionClient())
        message = "수소 액화 저장 관련 최근 2년간 논문, 세미나, 보고서를 검색해줘"
        with patch("runtime.agent_runtime.execute_research_action", return_value=[]) as run_tools:
            result = runtime.chat(message, "auto")

        self.assertEqual(result.route.agent, "research")
        self.assertEqual(result.route.search_mode, "DEEP_RESEARCH")
        run_tools.assert_called_once_with(
            "SEARCH_ACADEMIC", ("liquefied hydrogen storage papers 2024 2026",), "searxng", (), "web", "normal"
        )

    def test_deep_research_uses_evidence_analyst_critic_and_revision_passes(self) -> None:
        class PipelineClient(FakeClient):
            def post(self, url: str, json: dict[str, object]) -> FakeResponse:
                self.requests.append({"url": url, "json": json})
                responses = (
                    '{"search_mode":"DEEP_RESEARCH","queries":["researcher papers"]}',
                    '{"missing":[],"uncertain":[],"next_queries":[],"next_tools":[],"ready_to_answer":true,"entity_confidence":"HIGH"}',
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
            result = runtime.chat(
                "연구자 역량을 평가해줘", "auto", persistent_context="Project decision: pressure is 12 bar"
            )

        self.assertEqual(result.content, "final revision")
        self.assertEqual(len(runtime._client.requests), 5)
        analyst_input = runtime._client.requests[2]["json"]["messages"][1]["content"]
        critic_input = runtime._client.requests[3]["json"]["messages"][1]["content"]
        final_input = runtime._client.requests[4]["json"]["messages"][1]["content"]
        self.assertIn("Evidence Package", analyst_input)
        self.assertIn("Project decision: pressure is 12 bar", analyst_input)
        self.assertIn("Project decision: pressure is 12 bar", critic_input)
        self.assertIn("Project decision: pressure is 12 bar", final_input)
        self.assertIn("Analyst Draft", critic_input)
        self.assertIn("FACT, INFERENCE", analyst_input)
        self.assertIn("causal_chain", analyst_input)
        self.assertIn("BULL/BASE/BEAR", analyst_input)
        self.assertIn("do not delete a valid inference", critic_input)
        self.assertIn("Explicitly list every number", critic_input)
        self.assertIn("Facts are sourced; inferences are reasoned", final_input)
        self.assertIn("never apply it to an inference or forecast", final_input)
        self.assertIn("audit every number and named-company relationship", final_input)
        self.assertIn("qualitative scenarios unless sourced numbers exist", final_input)
        self.assertIn('"analysis_contract"', analyst_input)
        self.assertIn('"FIRST_ORDER_EFFECTS"', analyst_input)
        self.assertIn('"inference_object"', analyst_input)
        self.assertTrue(result.research["final_synthesis_executed"])
        self.assertEqual(result.research["state"], "COMPLETE")
        self.assertEqual(result.research["claim_taxonomy"], ("FACT", "INFERENCE", "FORECAST", "UNKNOWN"))

    def test_web_evidence_pack_assigns_analytical_roles_and_ids(self) -> None:
        sources = [
            {"title": "Official results", "url": "https://investor.example.com/results", "text": "Revenue increased."},
            {"title": "Supply chain", "url": "https://example.com/chain", "text": "Supplier capacity and pricing affect margin."},
            {"title": "Demand risk", "url": "https://example.com/risk", "text": "However, demand weakness is a risk."},
            {"title": "Market report", "url": "https://example.com/report", "text": "Independent market context."},
        ]

        package = AgentRuntime._evidence_package([{
            "name": "web_sources", "success": True, "output": json.dumps(sources), "error": None,
        }])

        self.assertEqual([source["evidence_id"] for source in package["sources"]], ["S1", "S2", "S3", "S4"])
        self.assertEqual(
            [source["evidence_role"] for source in package["sources"]],
            ["DIRECT", "STRUCTURAL", "CONTRADICTORY", "SUPPORTING"],
        )

    def test_evidence_pack_preserves_role_diversity_before_relevance_fill(self) -> None:
        sources = [
            {"title": f"Official {index}", "url": f"https://investor.example.com/{index}", "text": "Results", "relevance_score": 1 - index / 100}
            for index in range(8)
        ] + [
            {"title": "Value chain", "url": "https://industry.example.com/chain", "text": "Supplier capacity affects margin", "relevance_score": 0.5},
            {"title": "Counter case", "url": "https://market.example.com/risk", "text": "However demand weakness is a risk", "relevance_score": 0.4},
        ]

        package = AgentRuntime._evidence_package([{
            "name": "web_sources", "success": True, "output": json.dumps(sources), "error": None,
        }])
        roles = {source["evidence_role"] for source in package["sources"]}

        self.assertEqual(len(package["sources"]), 6)
        self.assertIn("DIRECT", roles)
        self.assertIn("STRUCTURAL", roles)
        self.assertIn("CONTRADICTORY", roles)

    def test_gap_analysis_allows_supported_inference_without_direct_conclusion(self) -> None:
        class GapClient(FakeClient):
            def post(self, url: str, json: dict[str, object]) -> FakeResponse:
                self.requests.append({"url": url, "json": json})
                response = FakeResponse()
                response.json = lambda: {"choices": [{"message": {"content": (
                    '{"missing":[],"uncertain":["valuation transmission assumption"],'
                    '"next_queries":[],"next_tools":[],"ready_to_answer":true,'
                    '"entity_confidence":"NOT_APPLICABLE"}'
                )}}], "usage": {}}  # type: ignore[method-assign]
                return response

        client = GapClient()
        runtime = AgentRuntime(client=client)
        question = "A 회사 실적이 B 산업에 어떤 영향을 줄까?"
        gap = runtime._research_gap(
            question, (question,), [], research_source_plan(question), "system", LatencyRecorder()
        )

        prompt = client.requests[0]["json"]["messages"][1]["content"]
        self.assertTrue(gap.ready_to_answer)
        self.assertIn("Absence of a source stating the final conclusion verbatim", prompt)
        self.assertIn("value-chain relationship", prompt)
        self.assertNotIn("Mark unavailable material NOT VERIFIED", prompt)

    def test_deep_research_bounds_large_evidence_and_revision_drafts(self) -> None:
        class LargePipelineClient(FakeClient):
            def post(self, url: str, json: dict[str, object]) -> FakeResponse:
                self.requests.append({"url": url, "json": json})
                responses = (
                    '{"search_mode":"DEEP_RESEARCH","queries":["researcher papers"]}',
                    '{"missing":[],"uncertain":[],"next_queries":[],"next_tools":[],"ready_to_answer":true,"entity_confidence":"HIGH"}',
                    "A" * 20_000,
                    "B" * 10_000,
                    "bounded final revision",
                )
                response = FakeResponse()
                response.json = lambda: {"choices": [{"message": {"content": responses[len(self.requests) - 1]}}], "usage": {}}  # type: ignore[method-assign]
                return response

        works = [
            {"title": f"Paper {index}", "doi": f"10.1000/{index}", "abstract": "X" * 10_000}
            for index in range(12)
        ]
        tools = [{
            "name": "academic_intelligence",
            "success": True,
            "output": json.dumps({
                "researcher": {"canonical_name": "Researcher", "identity_confidence": "HIGH"},
                "representative_papers": works,
            }),
            "error": None,
        }]
        runtime = AgentRuntime(client=LargePipelineClient())
        with patch("runtime.agent_runtime.run_agent_tools", return_value=tools):
            result = runtime.chat("연구자 역량을 평가해줘", "research")

        critic_input = runtime._client.requests[3]["json"]["messages"][1]["content"]
        final_input = runtime._client.requests[4]["json"]["messages"][1]["content"]
        self.assertLess(len(critic_input), 20_500)
        self.assertLess(len(final_input), 23_500)
        self.assertIn("truncated for context budget", critic_input)
        self.assertEqual(result.content, "bounded final revision")

    def test_bounded_evidence_keeps_official_details_after_navigation_text(self) -> None:
        package = {
            "sources": [{
                "title": "NVIDIA conference call",
                "url": "https://nvidianews.nvidia.com/news/example",
                "text": "Navigation " * 65 + "Official call: August 26 at 5 p.m. ET.",
            }]
        }

        bounded = AgentRuntime._bounded_evidence_json(package)

        self.assertIn("Official call: August 26 at 5 p.m. ET.", bounded)

    def test_market_synthesis_includes_utc_and_kst_research_timestamp(self) -> None:
        client = FakeClient()
        runtime = AgentRuntime(client=client)
        question = "NVIDIA 오늘 실적 발표와 시장 전망을 조사해줘"

        runtime._synthesize_research(
            question, [], "system", LatencyRecorder(), source_plan=research_source_plan(question)
        )

        analyst_input = client.requests[0]["json"]["messages"][1]["content"]
        self.assertIn('"research_as_of"', analyst_input)
        self.assertIn('"utc"', analyst_input)
        self.assertIn('"kst"', analyst_input)
        self.assertIn("+09:00", analyst_input)

    def test_direct_research_selection_still_classifies_deep_research(self) -> None:
        class DirectResearchClient(FakeClient):
            def post(self, url: str, json: dict[str, object]) -> FakeResponse:
                self.requests.append({"url": url, "json": json})
                responses = (
                    '{"search_mode":"NO_SEARCH","queries":[]}',
                    '{"missing":[],"uncertain":[],"next_queries":[],"next_tools":[],"ready_to_answer":true,"entity_confidence":"HIGH"}',
                    "analyst", "critic", "완료된 최종 답변",
                )
                response = FakeResponse()
                response.json = lambda: {"choices": [{"message": {"content": responses[len(self.requests) - 1]}}], "usage": {}}  # type: ignore[method-assign]
                return response

        runtime = AgentRuntime(client=DirectResearchClient())
        with patch("runtime.agent_runtime.run_agent_tools", return_value=[]):
            result = runtime.chat("연구자로서의 역량을 근거로 평가해줘", "research")

        self.assertEqual(result.route.search_mode, "DEEP_RESEARCH")
        self.assertEqual(result.content, "완료된 최종 답변")
        self.assertEqual(result.llm_calls[0]["purpose"], "research_mode_decision")

    def test_research_classifier_receives_project_context_for_reference_resolution(self) -> None:
        class ContextClassifierClient(FakeClient):
            def post(self, url: str, json: dict[str, object]) -> FakeResponse:
                self.requests.append({"url": url, "json": json})
                response = FakeResponse()
                response.json = lambda: {  # type: ignore[method-assign]
                    "choices": [{"message": {"content": '{"search_mode":"DEEP_RESEARCH","queries":["안호선 교수 연구 실적"]}'}}],
                    "usage": {},
                }
                return response

        client = ContextClassifierClient()
        runtime = AgentRuntime(client=client)

        decision = runtime._search_decision(
            "이 연구자가 왜 뛰어난지 근거를 찾아봐",
            persistent_context="Project subject: 안호선 교수",
            research_agent_selected=True,
        )

        classifier_input = client.requests[0]["json"]["messages"][1]["content"]
        self.assertEqual(decision.mode, "DEEP_RESEARCH")
        self.assertIn("Research agent selected: True", classifier_input)
        self.assertIn("Project subject: 안호선 교수", classifier_input)
        self.assertIn("not external evidence", classifier_input)

    def test_unready_gap_executes_followup_round_before_final_synthesis(self) -> None:
        class FollowupClient(FakeClient):
            def post(self, url: str, json: dict[str, object]) -> FakeResponse:
                self.requests.append({"url": url, "json": json})
                responses = (
                    '{"search_mode":"DEEP_RESEARCH","queries":["first query"]}',
                    '{"missing":["identity"],"uncertain":[],"next_queries":["second query"],"next_tools":["web_search"],"ready_to_answer":false,"entity_confidence":"UNRESOLVED"}',
                    '{"missing":[],"uncertain":[],"next_queries":[],"next_tools":[],"ready_to_answer":true,"entity_confidence":"HIGH"}',
                    "analyst", "critic", "최종 평가",
                )
                response = FakeResponse()
                response.json = lambda: {"choices": [{"message": {"content": responses[len(self.requests) - 1]}}], "usage": {}}  # type: ignore[method-assign]
                return response

        runtime = AgentRuntime(client=FollowupClient())
        with patch("runtime.agent_runtime.execute_research_action", return_value=[]) as run_tools:
            result = runtime.chat("안호선 교수의 연구 역량을 평가해줘", "auto")

        self.assertEqual(run_tools.call_count, 1)
        self.assertEqual(
            run_tools.call_args.args[1],
            ("second query",),
        )
        self.assertEqual(len(result.research["rounds"]), 2)
        self.assertFalse(result.research["rounds"][0]["ready_to_answer"])
        self.assertTrue(result.research["final_synthesis_executed"])
        self.assertEqual(result.research["state_history"][0], "PLANNING")
        self.assertIn("SEARCHING", result.research["state_history"])
        self.assertEqual(result.research["state_history"][-2:], ["SYNTHESIZING", "COMPLETE"])
        self.assertEqual(result.content, "최종 평가")

    def test_premature_final_decision_continues_with_requested_tool_action(self) -> None:
        class NextActionClient(FakeClient):
            def post(self, url: str, json: dict[str, object]) -> FakeResponse:
                self.requests.append({"url": url, "json": json})
                responses = (
                    '{"search_mode":"DEEP_RESEARCH","queries":["NVIDIA recent issues"]}',
                    '{"next_action":"FINAL_ANSWER","queries":[],"provider":"",'
                    '"unresolved_questions":["current developments"],"decision_summary":"More search is needed",'
                    '"ready_to_answer":false,"complexity":"SIMPLE","use_critic":false}',
                    '{"next_action":"SEARCH_WEB","queries":["NVIDIA recent issues"],"provider":"searxng",'
                    '"unresolved_questions":["current developments"],"decision_summary":"Find current evidence",'
                    '"ready_to_answer":false,"complexity":"SIMPLE","use_critic":false}',
                    '{"next_action":"FINAL_ANSWER","queries":[],"provider":"",'
                    '"unresolved_questions":[],"decision_summary":"Evidence is sufficient",'
                    '"ready_to_answer":true,"complexity":"SIMPLE","use_critic":false}',
                    "현재 근거에 따른 최종 답변",
                )
                response = FakeResponse()
                response.json = lambda: {"choices": [{"message": {"content": responses[len(self.requests) - 1]}}], "usage": {}}  # type: ignore[method-assign]
                return response

        runtime = AgentRuntime(client=NextActionClient())
        observation = [{"name": "web_search", "success": True, "output": "[]", "error": None}]
        with patch("runtime.agent_runtime.execute_research_action", return_value=observation) as execute:
            result = runtime.chat("Nvidia 최근 이슈 정리", "auto")

        execute.assert_called_once_with("SEARCH_WEB", ("NVIDIA recent issues",), "searxng", (), "web", "normal")
        self.assertEqual(result.content, "현재 근거에 따른 최종 답변")
        self.assertEqual([step["decision"] for step in result.research["rounds"]], [
            "FINAL_ANSWER", "SEARCH_WEB", "FINAL_ANSWER",
        ])
        self.assertFalse(result.research["rounds"][0]["ready_to_answer"])
        self.assertEqual(result.research["termination_reason"], "llm_evidence_sufficient")
        self.assertEqual(result.llm_calls[-1]["purpose"], "direct_research_synthesis")

    def test_invalid_gap_output_triggers_followup_instead_of_early_completion(self) -> None:
        class InvalidGapClient(FakeClient):
            def post(self, url: str, json: dict[str, object]) -> FakeResponse:
                self.requests.append({"url": url, "json": json})
                responses = (
                    '{"search_mode":"DEEP_RESEARCH","queries":["first query"]}',
                    "Let me search more specifically.",
                    '{"missing":[],"uncertain":[],"next_queries":[],"next_tools":[],'
                    '"ready_to_answer":true,"entity_confidence":"HIGH"}',
                    "analyst", "critic", "최종 평가",
                )
                response = FakeResponse()
                response.json = lambda: {"choices": [{"message": {"content": responses[len(self.requests) - 1]}}], "usage": {}}  # type: ignore[method-assign]
                return response

        runtime = AgentRuntime(client=InvalidGapClient())
        with patch("runtime.agent_runtime.execute_research_action", return_value=[]) as run_tools:
            result = runtime.chat("안호선 교수의 연구 역량을 평가해줘", "research")

        self.assertEqual(run_tools.call_count, 1)
        self.assertEqual(result.research["rounds"][0]["decision"], "SEARCH_WEB")
        self.assertFalse(result.research["rounds"][0]["ready_to_answer"])
        self.assertEqual(result.research["state"], "COMPLETE")
        self.assertTrue(result.research["final_synthesis_executed"])

    def test_intermediate_research_progress_cannot_be_final_response(self) -> None:
        class ProgressClient(FakeClient):
            def post(self, url: str, json: dict[str, object]) -> FakeResponse:
                self.requests.append({"url": url, "json": json})
                responses = (
                    '{"search_mode":"DEEP_RESEARCH","queries":["evidence"]}',
                    '{"missing":[],"uncertain":[],"next_queries":[],"next_tools":[],"ready_to_answer":true,"entity_confidence":"HIGH"}',
                    "analyst", "critic",
                    "I'll investigate this. Let me search more specifically.",
                    '{"next_action":"SEARCH_WEB","queries":["specific evidence"],"provider":"searxng",'
                    '"unresolved_questions":["specific evidence"],"decision_summary":"Run the needed search",'
                    '"ready_to_answer":false,"complexity":"SIMPLE","use_critic":false}',
                    '{"next_action":"FINAL_ANSWER","queries":[],"provider":"","unresolved_questions":[],'
                    '"decision_summary":"Evidence sufficient","ready_to_answer":true,"complexity":"SIMPLE","use_critic":false}',
                    "## 결론\n검증된 근거에 따른 최종 답변입니다.",
                )
                response = FakeResponse()
                response.json = lambda: {"choices": [{"message": {"content": responses[len(self.requests) - 1]}}], "usage": {}}  # type: ignore[method-assign]
                return response

        runtime = AgentRuntime(client=ProgressClient())
        with patch("runtime.agent_runtime.execute_research_action", return_value=[]) as execute:
            result = runtime.chat("근거를 찾아 평가해줘", "auto")

        self.assertNotIn("I'll investigate", result.content)
        execute.assert_called_once()
        self.assertEqual(result.llm_calls[-1]["purpose"], "direct_research_synthesis")
        self.assertTrue(result.research["final_synthesis_executed"])

    def test_repeated_research_progress_output_requires_another_action(self) -> None:
        class RepeatedProgressClient(FakeClient):
            def post(self, url: str, json: dict[str, object]) -> FakeResponse:
                self.requests.append({"url": url, "json": json})
                responses = (
                    '{"search_mode":"DEEP_RESEARCH","queries":["evidence"]}',
                    '{"missing":[],"uncertain":[],"next_queries":[],"next_tools":[],'
                    '"ready_to_answer":true,"entity_confidence":"HIGH"}',
                    "analyst", "critic",
                    "I'll investigate this. Let me search more specifically.",
                    '{"next_action":"SEARCH_WEB","queries":["another source"],"provider":"searxng",'
                    '"unresolved_questions":["another source"],"decision_summary":"Search another source",'
                    '"ready_to_answer":false,"complexity":"SIMPLE","use_critic":false}',
                    '{"next_action":"FINAL_ANSWER","queries":[],"provider":"","unresolved_questions":[],'
                    '"decision_summary":"Ready","ready_to_answer":true,"complexity":"SIMPLE","use_critic":false}',
                    "I need more information. I'll check another source.",
                    '{"next_action":"SEARCH_WEB","queries":["last verification"],"provider":"serper",'
                    '"unresolved_questions":["last verification"],"decision_summary":"Perform last verification",'
                    '"ready_to_answer":false,"complexity":"SIMPLE","use_critic":false}',
                    '{"next_action":"FINAL_ANSWER","queries":[],"provider":"","unresolved_questions":[],'
                    '"decision_summary":"Ready","ready_to_answer":true,"complexity":"SIMPLE","use_critic":false}',
                    "완료된 최종 답변",
                )
                response = FakeResponse()
                response.json = lambda: {"choices": [{"message": {"content": responses[len(self.requests) - 1]}}], "usage": {}}  # type: ignore[method-assign]
                return response

        runtime = AgentRuntime(client=RepeatedProgressClient())
        with patch("runtime.agent_runtime.execute_research_action", return_value=[]) as execute:
            result = runtime.chat("근거를 찾아 평가해줘", "research")

        self.assertEqual(execute.call_count, 2)
        self.assertEqual(result.content, "완료된 최종 답변")

    def test_general_and_project_chat_use_same_deep_research_runtime(self) -> None:
        class EquivalentClient(FakeClient):
            def post(self, url: str, json: dict[str, object]) -> FakeResponse:
                self.requests.append({"url": url, "json": json})
                responses = (
                    '{"search_mode":"DEEP_RESEARCH","queries":["researcher evidence"]}',
                    '{"missing":[],"uncertain":[],"next_queries":[],"next_tools":[],"ready_to_answer":true,"entity_confidence":"HIGH"}',
                    "analyst", "critic", "최종 답변",
                )
                response = FakeResponse()
                response.json = lambda: {"choices": [{"message": {"content": responses[len(self.requests) - 1]}}], "usage": {}}  # type: ignore[method-assign]
                return response

        question = "연구자에 대해서 찾아보고 연구자로서의 역량을 평가해줘"
        project_scope = object()
        with patch("runtime.agent_runtime.execute_research_action", return_value=[]) as run_tools:
            general = AgentRuntime(client=EquivalentClient()).chat(question, "auto")
            project = AgentRuntime(client=EquivalentClient()).chat(
                question, "auto", persistent_context="Prior project note", project_scope=project_scope  # type: ignore[arg-type]
            )

        self.assertEqual(general.route.search_mode, project.route.search_mode)
        self.assertEqual(general.content, project.content)
        self.assertEqual([call["purpose"] for call in general.llm_calls], [call["purpose"] for call in project.llm_calls])
        self.assertEqual(len(general.research["rounds"]), len(project.research["rounds"]))
        run_tools.assert_not_called()

    def test_korean_person_name_is_preserved_as_exact_query(self) -> None:
        queries = AgentRuntime._initial_research_queries(
            "안호선 교수가 왜 뛰어난지 근거를 찾아봐",
            ("Ahn Ho Seon publications",),
        )

        self.assertEqual(queries[0], '"안호선"')
        self.assertIn("안호선 교수가 왜 뛰어난지 근거를 찾아봐", queries)
        self.assertIn("Ahn Ho Seon publications", queries)

    def test_generic_researcher_phrases_are_not_treated_as_korean_names(self) -> None:
        overseas = AgentRuntime._initial_research_queries(
            "Geoffrey Hinton이 왜 유명한 해외 연구자인지 평가해줘", ()
        )
        sparse = AgentRuntime._initial_research_queries("정보가 거의 없는 연구자를 조사해줘", ())

        self.assertNotIn('"해외"', overseas)
        self.assertNotIn('"없는"', sparse)

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

        self.assertEqual(client.requests[-1]["json"]["max_tokens"], 6144)
        self.assertIn("truncated", result.content)

    def test_brave_search_limits_quick_results(self) -> None:
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "test-key"}), patch("runtime.web_search.httpx.get", return_value=BraveResponse()) as get:
            results = search("latest example", "QUICK_SEARCH")

        self.assertEqual(len(results), 5)
        self.assertEqual(get.call_args.kwargs["params"]["count"], 5)

    def test_visual_search_is_bounded_and_uses_strict_safesearch(self) -> None:
        with patch.dict(os.environ, {"BRAVE_SEARCH_API_KEY": "test-key"}), patch(
            "runtime.web_search.httpx.get", return_value=BraveImageResponse()
        ) as get:
            results = visual_search("anime pose reference", 3)

        self.assertEqual(len(results), 3)
        self.assertEqual(get.call_args.args[0], "https://api.search.brave.com/res/v1/images/search")
        self.assertEqual(get.call_args.kwargs["params"]["safesearch"], "strict")
        self.assertEqual(get.call_args.kwargs["params"]["count"], 3)

    def test_korean_deep_research_uses_naver_and_reddit(self) -> None:
        responses = [NaverResponse(), RedditResponse()]
        with patch.dict(os.environ, {
            "SEARXNG_URL": "", "SERPER_API_KEY": "", "BRAVE_SEARCH_API_KEY": "",
            "NAVER_SEARCH_CLIENT_ID": "naver-id", "NAVER_SEARCH_CLIENT_SECRET": "naver-secret",
        }), patch("runtime.web_search.httpx.get", side_effect=responses) as get:
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
        self.assertIn("Verified source text with substantive details", sources[0]["text"])
        self.assertNotIn("secret", sources[0]["text"])

    def test_deep_research_rejects_insufficient_page_text(self) -> None:
        results = [{"title": "Thin page", "url": "https://example.com/thin", "description": "Snippet"}]
        response = SourceResponse()
        response.text = "<html><body>Brief navigation shell.</body></html>"
        with patch("runtime.web_search.socket.getaddrinfo", return_value=[(2, 1, 6, "", ("93.184.216.34", 0))]), patch(
            "runtime.web_search.httpx.get", return_value=response
        ):
            sources, metrics = fetch_sources(results, include_metrics=True)

        self.assertEqual(sources, [])
        self.assertEqual(metrics[0]["failure_reason"], "insufficient_extracted_text")

    def test_deep_research_reads_body_after_large_page_head(self) -> None:
        results = [{"title": "Official newsroom", "url": "https://example.com/news", "description": "Snippet"}]
        response = SourceResponse()
        response.text = (
            "<html><head><script>" + "ignored metadata " * 2_000 + "</script></head><body><article>"
            + "Official earnings date, conference call time, revenue, and guidance details. " * 5
            + "</article></body></html>"
        )
        with patch("runtime.web_search.socket.getaddrinfo", return_value=[(2, 1, 6, "", ("93.184.216.34", 0))]), patch(
            "runtime.web_search.httpx.get", return_value=response
        ):
            sources = fetch_sources(results)

        self.assertIn("Official earnings date", sources[0]["text"])
        self.assertNotIn("ignored metadata", sources[0]["text"])

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

    def test_researcher_query_preserves_native_name_and_extracts_person_alias(self) -> None:
        query = _researcher_query(("안호선 교수 연구 실적", "Ho Seon Ahn Incheon National University"))

        self.assertEqual(query, "안호선 교수 연구 실적\nAcademic alias: Ho Seon Ahn")

    def test_researcher_query_rejects_institution_as_person_alias(self) -> None:
        query = _researcher_query(("안호선 교수 연구 실적", "Incheon National University mechanical engineering"))

        self.assertEqual(query, "안호선 교수 연구 실적")

    def test_llm_resolves_publication_name_from_public_web_evidence(self) -> None:
        class IdentityClient(FakeClient):
            def post(self, url: str, json: dict[str, object]) -> FakeResponse:
                self.requests.append({"url": url, "json": json})
                response = FakeResponse()
                response.json = lambda: {  # type: ignore[method-assign]
                    "choices": [{"message": {"content": json_module.dumps({
                        "canonical_name": "안호선",
                        "publication_name": "Ho Seon Ahn",
                        "affiliation": "Incheon National University",
                        "confidence": "HIGH",
                    })}}],
                    "usage": {},
                }
                return response

        json_module = json
        runtime = AgentRuntime(client=IdentityClient())
        resolved = runtime._resolve_researcher_identity_query(
            "안호선 교수의 연구역량을 조사해봐",
            ('"안호선"', "Incheon National University mechanical engineering"),
            json.dumps([{"title": "안호선 교수 연구실", "description": "Ho Seon Ahn, Incheon National University"}]),
            "system",
            LatencyRecorder(),
        )

        self.assertIn("안호선 교수", resolved)
        self.assertIn("Academic alias: Ho Seon Ahn", resolved)
        self.assertIn("Affiliation hint: Incheon National University", resolved)

    def test_korean_romanization_adds_compact_and_syllable_publication_names(self) -> None:
        aliases = AgentRuntime._romanization_aliases("안호선", ["Ho-Sun Ahn"])

        self.assertEqual(aliases, ("Hoseon Ahn", "Ho Seon Ahn"))

    def test_s2_author_queries_extracts_and_reorders_hyphenated_professor_name(self) -> None:
        queries = _s2_author_queries("Incheon University Ho-Sun Ahn professor research papers")

        self.assertEqual(queries, ("Ho Sun Ahn", "Sun Ho Ahn"))

    def test_s2_author_queries_extracts_korean_professor_name(self) -> None:
        self.assertEqual(_s2_author_queries("안호선교수 연구 역량"), ("안호선",))

    def test_s2_author_queries_extracts_romanized_name_before_korean_particle(self) -> None:
        self.assertEqual(
            _s2_author_queries("Geoffrey Hinton이 왜 유명한 해외 연구자인지 평가해줘"),
            ("Geoffrey Hinton",),
        )

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

    def test_deep_research_researcher_query_uses_multi_source_orchestrator(self) -> None:
        intelligence = {
            "researcher": {"identity_confidence": "MEDIUM", "identity_sources": ["openalex", "semantic_scholar"]},
            "source_status": {"scopus": "UNAVAILABLE", "web_of_science": "UNAVAILABLE", "openalex": "AVAILABLE_FULL"},
            "coverage": {"openalex": {"publication_count": 8}}, "conflicts": [],
            "merged_publication_count": 8, "representative_papers": [], "cache_hit": False,
            "selection_policy": {"providers_called": ["openalex", "semantic_scholar", "orcid", "crossref"]},
        }
        with patch(
            "runtime.tool_registry._web_search", return_value=ToolResult("web_search", True, "[]", None, 0)
        ), patch("runtime.tool_registry._web_sources", return_value=ToolResult("web_sources", False, "", "empty", 0)), patch(
            "runtime.tool_registry.academic_intelligence", return_value=intelligence
        ) as orchestrator:
            results = _research_tools(("안호선교수 연구 역량을 평가해줘",), "DEEP_RESEARCH", False)

        self.assertEqual(results[-1].name, "academic_intelligence")
        self.assertTrue(results[-1].success)
        self.assertEqual(results[-1].details["execution"], "parallel")
        self.assertIn("orcid", results[-1].details["providers_called"])
        orchestrator.assert_called_once()

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