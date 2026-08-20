# Tests

Add focused checks for agent clients, configuration, and serving integration in
this directory. The existing `scripts/smoke-test.sh` validates the live vLLM API.

`test_coding_agent_fixture.py` is a small, safe workspace task used to evaluate
the Coding Agent's inspect-edit-test-diff loop:

```bash
python3 -m unittest tests/test_coding_agent_fixture.py
```

`test_agent_artifacts.py` checks the reusable Coding Agent instruction and
endpoint-profile contract.

`test_gpu_documentation.py` prevents obsolete GPU assumptions from returning to
the current RTX PRO 6000 Blackwell deployment documentation.

`test_specialist_agents.py` checks Research and Server role boundaries, their
evaluation task sets, and Main Agent routing.

`test_web_runtime.py` verifies Main routing, short-term session history,
read-only tools, and FastAPI's runtime integration with a mock Qwen client.