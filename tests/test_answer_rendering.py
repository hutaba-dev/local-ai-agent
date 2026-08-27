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
    def render(self, markdown: str) -> str:
        source = SCRIPT.read_text(encoding="utf-8")
        start = source.index("function escapeHtml")
        end = source.index("function enhanceResearchMessage")
        renderer = source[start:end]
        program = f"{renderer}\nprocess.stdout.write(markdown({json.dumps(markdown)}));"
        return subprocess.run(
            ["node", "-e", program], check=True, capture_output=True, text=True
        ).stdout

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


if __name__ == "__main__":
    unittest.main()
