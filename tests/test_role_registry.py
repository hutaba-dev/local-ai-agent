from __future__ import annotations

import unittest
from pathlib import Path

from runtime.capability_registry import PermissionClass
from runtime.role_registry import client_default_role, get_role, registry_snapshot, selectable_roles


class RoleRegistryTests(unittest.TestCase):
    def test_vscode_agent_uses_the_central_kim_coding_role(self) -> None:
        agent = (Path(__file__).resolve().parents[1] / "agents/coding/vscode-coding.agent.md").read_text()

        self.assertIn('model: "qwen3.8-27b"', agent)
        self.assertIn("tools: [read, search, edit, execute, todo]", agent)
        self.assertIn("KIM operating with Coding as the primary role", agent)
        self.assertIn("[Coding Role instructions](instructions.md)", agent)

    def test_kim_roles_share_the_brain_identity_without_becoming_models(self) -> None:
        self.assertEqual(get_role("main").id, "secretary")
        self.assertEqual(get_role("coding").id, "coder")
        self.assertEqual(get_role("research").id, "researcher")
        self.assertTrue(all(role.name.startswith("KIM / ") for role in selectable_roles()))
        self.assertTrue(all("model" not in item for item in registry_snapshot()))

    def test_coder_has_safe_defaults_and_optional_cross_role_capabilities(self) -> None:
        coder = get_role("coder")

        self.assertEqual(coder.default_capabilities, (
            "workspace_read", "workspace_search", "workspace_edit", "execute",
        ))
        self.assertTrue({"workspace", "documentation", "git", "github", "web"} <= set(coder.preferred_capabilities))
        self.assertEqual(set(coder.permission_policy), {
            PermissionClass.READ,
            PermissionClass.READ_PROJECT,
            PermissionClass.WRITE_WORKSPACE,
            PermissionClass.EXECUTE_SAFE,
        })
        self.assertNotIn(PermissionClass.WRITE_REPOSITORY, coder.permission_policy)
        self.assertNotIn(PermissionClass.DESTRUCTIVE, coder.permission_policy)

    def test_researcher_can_prefer_authoritative_github_evidence(self) -> None:
        researcher = get_role("researcher")

        self.assertIn("github", researcher.preferred_capabilities)
        self.assertIn(PermissionClass.READ, researcher.permission_policy)

    def test_vscode_defaults_to_coder_while_other_clients_keep_secretary(self) -> None:
        self.assertEqual(client_default_role("vscode").id, "coder")
        self.assertEqual(client_default_role("web").id, "secretary")
        self.assertEqual(client_default_role("unknown").id, "secretary")

    def test_all_instruction_files_are_loadable(self) -> None:
        for role in selectable_roles():
            self.assertTrue(role.instructions().strip(), role.id)


if __name__ == "__main__":
    unittest.main()