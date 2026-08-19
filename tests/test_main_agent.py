import unittest

from agents.common.memory import InMemoryMemoryStore
from agents.main.delegation import delegate_to_coding


class MainToCodingDelegationTests(unittest.TestCase):
    def test_main_creates_a_complete_coding_handoff(self) -> None:
        handoff = delegate_to_coding(
            goal="Add a focused test for the configuration parser.",
            constraints=("Do not change public APIs.",),
            acceptance_checks=("Run the focused test.", "Review git diff."),
            commit_requested=False,
        )

        self.assertEqual(handoff.target, "coding")
        self.assertFalse(handoff.commit_requested)
        self.assertTrue(handoff.goal)
        self.assertTrue(handoff.acceptance_checks)

    def test_long_term_memory_requires_explicit_safe_save_and_supports_search_delete(self) -> None:
        store = InMemoryMemoryStore()
        record = store.save(
            kind="preference",
            content="Prefer focused tests before a full suite.",
            tags=("testing", "workflow"),
        )

        self.assertEqual(store.search("focused testing"), [record])
        self.assertTrue(store.delete(record.id))
        self.assertEqual(store.search("focused testing"), [])

    def test_long_term_memory_rejects_potential_secrets(self) -> None:
        with self.assertRaisesRegex(ValueError, "sensitive"):
            InMemoryMemoryStore().save(
                kind="workflow",
                content="api_key=not-a-real-value",
                tags=("unsafe",),
            )


if __name__ == "__main__":
    unittest.main()