import sys
import unittest
from pathlib import Path


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "coding-agent-workspace"
sys.path.insert(0, str(FIXTURE_ROOT))

from text_metrics import count_nonempty_lines


class TextMetricsTests(unittest.TestCase):
    def test_counts_only_nonempty_lines(self) -> None:
        self.assertEqual(count_nonempty_lines("first\n\n  \nsecond\n"), 2)

    def test_empty_text_has_no_nonempty_lines(self) -> None:
        self.assertEqual(count_nonempty_lines("\n \n\t"), 0)


if __name__ == "__main__":
    unittest.main()