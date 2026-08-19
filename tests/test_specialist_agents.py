import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class SpecialistAgentTests(unittest.TestCase):
    def test_research_agent_requires_citations_and_source_interpretation_split(self) -> None:
        instructions = (REPO_ROOT / "agents/research/instructions.md").read_text()

        self.assertIn("**Source**", instructions)
        self.assertIn("**Interpretation**", instructions)
        self.assertIn("citations", instructions)
        self.assertIn("Do not invent citations", instructions)

    def test_server_agent_requires_human_approval_for_privileged_changes(self) -> None:
        instructions = (REPO_ROOT / "agents/server/instructions.md").read_text()

        self.assertIn("Human approval is mandatory", instructions)
        self.assertIn("**any** `sudo` command", instructions)
        self.assertIn("systemctl start/stop/restart/enable/disable", instructions)
        self.assertIn("BatchMode=yes", instructions)
        self.assertIn("local `allowed-ssh-hosts`", instructions)

    def test_each_specialist_has_at_least_five_evaluation_tasks(self) -> None:
        for role in ("research", "server"):
            tasks = (REPO_ROOT / f"agents/{role}/evaluation-tasks.md").read_text()
            task_count = sum(1 for line in tasks.splitlines() if line[:2] in {"1.", "2.", "3.", "4.", "5."})
            self.assertGreaterEqual(task_count, 5, role)

    def test_main_routes_to_both_specialists(self) -> None:
        instructions = (REPO_ROOT / "agents/main/instructions.md").read_text()

        self.assertIn("Research Agent", instructions)
        self.assertIn("Server Agent", instructions)
        self.assertIn("human approval", instructions)


if __name__ == "__main__":
    unittest.main()