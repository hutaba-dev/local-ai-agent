import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class AgentArtifactTests(unittest.TestCase):
    def test_coding_agent_inherits_the_common_constitution(self) -> None:
        instructions = (REPO_ROOT / "agents/coding/instructions.md").read_text()
        self.assertIn("../common/constitution.md", instructions)
        self.assertIn("git diff", instructions)
        self.assertIn("Do not declare", instructions)

    def test_endpoint_profile_uses_the_local_qwen_api(self) -> None:
        profile = (REPO_ROOT / "agents/coding/qwen-openai.env.example").read_text()
        self.assertIn("OPENAI_BASE_URL=http://127.0.0.1:8000/v1", profile)
        self.assertIn("OPENAI_MODEL=qwen3.8-27b", profile)
        self.assertIn("<LOCAL_VLLM_API_KEY>", profile)

    def test_constitution_blocks_unsafe_automatic_operations(self) -> None:
        constitution = (REPO_ROOT / "agents/common/constitution.md").read_text()
        for principle in (
            "Observe Before Acting",
            "Use Tools When Reality Can Be Checked",
            "Make The Smallest Justified Change",
            "Verify Before Declaring Success",
            "Learn From Environment Feedback",
            "Respect Permissions",
            "Protect Secrets And Personal Data",
            "Keep Changes Reproducible",
            "Use Git As The Audit Trail",
            "Separate Fact, Inference, And Uncertainty",
        ):
            self.assertIn(principle, constitution)
        for restricted_operation in ("rm -rf", "driver 또는 kernel", "force push"):
            self.assertIn(restricted_operation, constitution)


if __name__ == "__main__":
    unittest.main()