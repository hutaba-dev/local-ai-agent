# Coding Agent

Coding is a reusable specialist role used by the shared KIM/Qwen brain. It is
not a separate model or server. VS Code is one client and defaults to this role;
the Web runtime can select it while retaining the same central model.

| Concern | Location |
| --- | --- |
| Common safety and reporting rules | [../common/constitution.md](../common/constitution.md) |
| Coding workflow | [instructions.md](instructions.md) |
| Non-secret endpoint template | [qwen-openai.env.example](qwen-openai.env.example) |

The stable default set is workspace read/search/edit and safe execution.
Documentation, read-only Git/GitHub, Project Knowledge, and Web Research are
discovered on demand under host policy. Research, Server, and Secretary roles
reuse the common constitution and the same model.