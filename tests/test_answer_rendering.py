from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "web" / "static" / "app.js"


@unittest.skipUnless(shutil.which("node"), "Node.js is required for renderer tests")
class AnswerRenderingTests(unittest.TestCase):
    def render(self, markdown: str, sources: list[dict[str, str]] | None = None) -> str:
        source = SCRIPT.read_text(encoding="utf-8")
        start = source.index("function escapeHtml")
        end = source.index("function renderResearchAnswer")
        renderer = source[start:end]
        program = (
            f"{renderer}\nprocess.stdout.write(markdown("
            f"{json.dumps(markdown)}, {json.dumps(sources or [])}));"
        )
        return subprocess.run(
            ["node", "-e", program], check=True, capture_output=True, text=True
        ).stdout

    def present_chat_response(self, payload: dict[str, object]) -> dict[str, object]:
        source = SCRIPT.read_text(encoding="utf-8")
        start = source.index("function chatResponsePresentation")
        end = source.index("function addMessageAttachments", start)
        presenter = source[start:end]
        program = (
            f"{presenter}\nprocess.stdout.write(JSON.stringify("
            f"chatResponsePresentation({json.dumps(payload)})));"
        )
        output = subprocess.run(
            ["node", "-e", program], check=True, capture_output=True, text=True
        ).stdout
        return json.loads(output)

    def test_project_clarification_without_activity_is_renderable(self) -> None:
        presentation = self.present_chat_response({
            "content": "현재 연결된 Project가 없습니다. 어느 Project에 저장할까요?",
            "activity": None,
            "project_write": {"status": "PROJECT_NOT_SELECTED"},
        })

        self.assertEqual(presentation["label"], "Assistant")
        self.assertIsNone(presentation["activity"])
        self.assertIn("어느 Project", presentation["content"])

    def test_normal_research_response_preserves_routed_agent(self) -> None:
        presentation = self.present_chat_response({
            "content": "Research result",
            "activity": {"routed_agent": "research"},
            "research_result": {"body_markdown": "# Research result"},
        })

        self.assertEqual(presentation["label"], "research")
        self.assertEqual(presentation["activity"]["routed_agent"], "research")
        self.assertEqual(presentation["researchResult"]["body_markdown"], "# Research result")

    def test_canonical_response_state_matrix_is_renderable(self) -> None:
        fixtures = {
            "A research": {
                "content": "Research result",
                "activity": {"routed_agent": "research"},
                "research_result": {"body_markdown": "# Result"},
            },
            "B null route": {"content": "Direct answer", "activity": {"routed_agent": None}},
            "C null activity": {"content": "Plain answer", "activity": None},
            "D successful action": {
                "content": "Project created",
                "activity": {},
                "project_action": {"success": True, "status": "AVAILABLE"},
            },
            "E clarification": {
                "content": "Project 이름을 알려주세요.",
                "activity": None,
                "project_action": {"success": False, "status": "PROJECT_NAME_REQUIRED"},
            },
            "F partial success": {
                "content": "Saved with a warning",
                "activity": None,
                "project_write": {"success": False, "status": "PARTIAL_SUCCESS"},
            },
        }

        for name, payload in fixtures.items():
            with self.subTest(name=name):
                presentation = self.present_chat_response(payload)
                self.assertEqual(presentation["content"], payload["content"])
                self.assertIn(presentation["label"], {"Assistant", "research"})
                self.assertIn("activity", presentation)

    def test_malformed_chat_response_without_content_is_rejected(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        start = source.index("function chatResponsePresentation")
        end = source.index("function addMessageAttachments", start)
        presenter = source[start:end]
        program = f"{presenter}\nchatResponsePresentation({{activity: null}});"

        result = subprocess.run(["node", "-e", program], capture_output=True, text=True)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Invalid chat response: content is missing", result.stderr)

    def test_research_markdown_renders_sections_table_sources_and_statuses(self) -> None:
        output = self.render(
            "# NVIDIA earnings\n\nSummary with VERIFIED evidence.\n\n"
            "| Metric | Value | Source |\n|---|---|---|\n"
            "| Revenue | $96.2B | [Yahoo Finance](https://finance.yahoo.com/markets/stocks/example) |\n\n"
            "## Risks\n\nNOT VERIFIED guidance and PARTIALLY VERIFIED estimates."
        )

        self.assertIn('<div class="table-scroll"><table>', output)
        self.assertIn('>Yahoo Finance</a>', output)
        self.assertNotIn("https://finance.yahoo.com/markets", output.split(">Yahoo Finance</a>")[1])
        self.assertIn('class="verification-badge verified"', output)
        self.assertIn('class="verification-badge partial"', output)
        self.assertIn('class="verification-badge unverified"', output)

    def test_general_markdown_keeps_bullets_and_clickable_named_link(self) -> None:
        output = self.render("- Read [Reuters](https://www.reuters.com/world/example)\n- Compare results")

        self.assertIn("<ul>", output)
        self.assertIn('target="_blank" rel="noopener noreferrer"', output)
        self.assertIn(">Reuters</a>", output)

    def test_raw_url_uses_domain_label_and_hides_path(self) -> None:
        output = self.render("See https://investor.nvidia.com/financial-info/quarterly-results/default.aspx")

        self.assertIn(">NVIDIA IR</a>", output)
        self.assertNotIn(">https://", output)

    def test_unsafe_markdown_link_does_not_create_anchor(self) -> None:
        output = self.render("[unsafe](javascript:alert(1))")

        self.assertNotIn("<a ", output)
        self.assertNotIn("javascript:", output)

    def test_code_blocks_are_escaped_and_not_linkified(self) -> None:
        output = self.render("```html\n<script>alert('x')</script> https://example.com/path\n```")

        self.assertIn("&lt;script&gt;", output)
        self.assertNotIn('class="external-link"', output)

    def test_production_research_fixture_renders_headings_annotations_and_citations(self) -> None:
        output = self.render(
            "NVIDIA 최신 이슈 요약\n\n"
            "### 1. 핵심 이슈\n**FACT:** 주요 발표가 있었다. [S2, S4]\n\n"
            "### 2. 기타 NVIDIA 관련 이슈\n**UNKNOWN:** 추가 확인이 필요하다.\n\n"
            "### 3. 한계 및 불확실성\n*본문과 출처를 구분한다.*",
            [
                {"id": "S2", "title": "NVIDIA report", "domain": "reuters.com", "url": "https://reuters.com/example"},
                {"id": "S4", "title": "Industry analysis", "domain": "example.com", "url": "https://example.com/report"},
            ],
        )

        self.assertIn("<h3>1. 핵심 이슈</h3>", output)
        self.assertNotIn("###", output)
        self.assertIn('class="verification-badge fact"', output)
        self.assertIn('class="verification-badge unknown"', output)
        self.assertIn('class="external-link citation-chip"', output)
        self.assertIn('href="https://reuters.com/example"', output)
        self.assertNotIn("[S2, S4]", output)
        self.assertNotIn(">https://", output)

    def test_all_research_entry_points_use_canonical_renderer(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("function renderResearchAnswer(article, researchResult = {})", source)
        self.assertIn("const normalizedResearch = researchResult || activity?.research?.result || null", source)
        self.assertIn("markdown(body, normalizedResearch?.sources || [])", source)
        self.assertIn("renderResearchAnswer(article, normalizedResearch || {})", source)
        self.assertIn("message.research_result", source)
        self.assertNotIn("enhanceResearchMessage", source)

    def test_markdown_fallback_supports_standard_report_elements(self) -> None:
        output = self.render(
            "# Title\n\n**bold** and *italic* with [named link](https://example.com/page).\n\n"
            "> quoted evidence\n\n- first\n- second\n\n1. one\n2. two\n\n"
            "| A | B |\n|---|---|\n| x | `code` |\n\n```text\nblock code\n```"
        )

        for expected in ("<h1>Title</h1>", "<strong>bold</strong>", "<em>italic</em>", "<blockquote>", "<ul>", "<ol>", "<table>", "<code>code</code>", "<pre><code>block code</code></pre>"):
            self.assertIn(expected, output)


if __name__ == "__main__":
    unittest.main()
