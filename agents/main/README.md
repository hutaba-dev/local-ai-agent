# Main Agent

The Main Agent is the default user-facing agent and provides normal conversation
and secretary-style coordination. It uses the Qwen OpenAI-compatible endpoint
through the agent host configuration, but its role instructions remain separate
from model configuration.

It delegates workspace changes to the Coding Agent with the typed handoff in
[delegation.py](delegation.py). Common safety and memory rules are shared from
`agents/common/`, so future Research and Server roles reuse the same boundaries.