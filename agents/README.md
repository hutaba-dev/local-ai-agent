# Agents

Agent orchestration profiles and non-secret client configuration belong here.
They communicate with the vLLM OpenAI-compatible endpoint and do not own the
serving process.

All agent roles inherit [common/constitution.md](common/constitution.md). The
first implementation is documented in [coding/README.md](coding/README.md).
The Main Agent is the default user-facing coordinator; see
[main/README.md](main/README.md). Memory rules are shared in
[common/memory-policy.md](common/memory-policy.md).

Specialists are [research/README.md](research/README.md) for evidence and data
work, and [server/README.md](server/README.md) for approved server diagnostics
and operations.