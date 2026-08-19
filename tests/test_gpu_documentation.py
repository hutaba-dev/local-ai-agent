import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = (
    "README.md",
    "docs/decisions.md",
    "docs/environment-baseline.md",
    "docs/model-decision.md",
    "docs/model-serving.md",
)


class GpuDocumentationTests(unittest.TestCase):
    def test_current_hardware_is_consistently_documented(self) -> None:
        combined = "\n".join((REPO_ROOT / path).read_text() for path in DOCUMENTS)

        self.assertIn("RTX PRO 6000 Blackwell", combined)
        self.assertIn("96 GiB", combined)
        self.assertNotIn("A" + "6000", combined)
        self.assertNotIn(f"{4}{8} GiB", combined)
        self.assertNotIn(f"{4}{8}GB", combined)


if __name__ == "__main__":
    unittest.main()