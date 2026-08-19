# Coding Agent

This is the first reusable workspace agent role. It uses Qwen through the local
OpenAI-compatible endpoint while keeping model configuration separate from the
role instructions.

| Concern | Location |
| --- | --- |
| Common safety and reporting rules | [../common/constitution.md](../common/constitution.md) |
| Coding workflow | [instructions.md](instructions.md) |
| Non-secret endpoint template | [qwen-openai.env.example](qwen-openai.env.example) |

The required tool set is file listing/search/read, file edit/write, terminal,
Python, Git status/diff, and a test runner. Web and MCP access remain opt-in
permissions. Research, Server, and Secretary roles should reuse the common
constitution and add only their role-specific workflow.