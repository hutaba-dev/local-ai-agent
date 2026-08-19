# Coding Agent Evaluation

## Scope

The first Coding Agent is an instruction bundle, not a new autonomous runtime.
It is configured to use the current Qwen OpenAI-compatible endpoint through a
local, ignored configuration file. Its reusable behavior is defined by:

- [agents/common/constitution.md](../agents/common/constitution.md)
- [agents/coding/instructions.md](../agents/coding/instructions.md)
- [agents/coding/qwen-openai.env.example](../agents/coding/qwen-openai.env.example)

## End-To-End Fixture Run

The safe fixture at `tests/fixtures/coding-agent-workspace/text_metrics.py` was
used to exercise the required loop without changing serving configuration or
production files.

| Workflow step | Evidence |
| --- | --- |
| Confirm goal | Implement a function that counts lines with non-whitespace text. |
| Explore and read | Located the fixture workspace and its target module under `tests/fixtures/coding-agent-workspace/`. |
| Plan | Use `str.splitlines()` and test each line after `strip()`. |
| Minimal edit | Added `count_nonempty_lines(text: str) -> int` in the fixture module. |
| Validate | `python3 -m unittest tests/test_coding_agent_fixture.py` passed: 2 tests. |
| Review | `git status`, `git diff --check`, and the relevant diff are required by the Coding Agent instructions. |
| Report | This document records the validation and the commit message is proposed before commit. |

The fixture checks mixed nonempty, blank, and whitespace-only lines. It gives
future agent-client implementations a low-risk workspace task that requires
investigation, a source edit, and executable verification.

## Artifact Contract Validation

Run the contract checks with:

```bash
python3 -m unittest tests/test_agent_artifacts.py
```

They assert that the Coding Agent inherits the common constitution, uses the
local Qwen endpoint template, and records the required destructive-operation
boundaries. The endpoint itself remains validated by `./scripts/healthcheck.sh`
and `./scripts/smoke-test.sh`.

## Result

The evaluation passed on 2026-08-20. The agent role has separate model
configuration, an inspect-edit-test-diff workflow, explicit approval boundaries,
and a reusable common constitution for future Research, Server, and Secretary
roles.