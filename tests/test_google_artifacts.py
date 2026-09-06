from __future__ import annotations

import unittest

from runtime.google_artifacts import TabularDataset, render_markdown_report, sanitize_report


class GoogleArtifactContractTests(unittest.TestCase):
    def test_markdown_becomes_native_format_metadata_without_raw_markers(self) -> None:
        rendered = render_markdown_report(
            "# Report\n\n## Findings\nA **material** result.\n\n- First\n1. Second\n\n"
            "| Company | Value |\n|---|---:|\n| Alpha | 100 |"
        )

        self.assertEqual([item.style for item in rendered.paragraphs[:2]], ["HEADING_1", "HEADING_2"])
        self.assertEqual([item.list_kind for item in rendered.paragraphs if item.list_kind], ["bullet", "number"])
        self.assertEqual(len(rendered.bold), 1)
        self.assertEqual(len(rendered.tables), 1)
        self.assertNotIn("#", rendered.text)
        self.assertNotIn("**", rendered.text)
        self.assertNotIn("|---", rendered.text)

    def test_operational_agent_messages_are_removed(self) -> None:
        report = sanitize_report(
            "# Analysis\nVerified result.\nGoogle Docs를 만들 수 없습니다.\n"
            "Sheets API가 없어 수행할 수 없습니다.\n## Conclusion\nSupported conclusion."
        )
        self.assertIn("Verified result", report)
        self.assertIn("Supported conclusion", report)
        self.assertNotIn("만들 수 없습니다", report)
        self.assertNotIn("API가 없어", report)

    def test_typed_dataset_preserves_numbers_nulls_and_evidence(self) -> None:
        dataset = TabularDataset.from_mapping({
            "title": "Capacity",
            "columns": ["Supplier", "Capacity", "Evidence"],
            "column_types": ["text", "integer", "text"],
            "units": [None, "wafers/month", None],
            "sources": ["S1"],
            "rows": [["Alpha", 100000, "S1"], ["Beta", 80000, "S1"], ["Gamma", None, "UNKNOWN"]],
            "chart_candidates": [{
                "title": "Capacity by supplier", "chart_type": "BAR",
                "category_column": 0, "series_columns": [1],
            }],
        })
        self.assertIsInstance(dataset.rows[0][1], int)
        self.assertIsNone(dataset.rows[2][1])
        self.assertEqual(dataset.rows[0][2], "S1")
        self.assertIsNotNone(dataset.valid_chart())

    def test_text_only_or_unknown_dataset_has_no_valid_chart(self) -> None:
        dataset = TabularDataset.from_mapping({
            "title": "Statuses",
            "columns": ["Supplier", "Status", "Evidence"],
            "column_types": ["text", "text", "text"],
            "sources": ["S1"],
            "rows": [["Alpha", "Likely", "S1"], ["Beta", "UNKNOWN", "S1"]],
            "chart_candidates": [{
                "title": "Status", "chart_type": "BAR", "category_column": 0, "series_columns": [1],
            }],
        })
        self.assertIsNone(dataset.valid_chart())

    def test_numeric_columns_reject_strings_and_unknown_zero_conversion(self) -> None:
        base = {
            "title": "Invalid",
            "columns": ["Supplier", "Capacity", "Evidence"],
            "column_types": ["text", "integer", "text"],
            "sources": ["S1"],
            "chart_candidates": [],
        }
        for invalid in ("100,000 (S1)", "UNKNOWN", "0"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    TabularDataset.from_mapping({**base, "rows": [["A", invalid, "S1"], ["B", None, "UNKNOWN"]]})


if __name__ == "__main__":
    unittest.main()