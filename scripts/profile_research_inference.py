#!/usr/bin/env python3
"""Profile model calls made by one unchanged Deep Research request."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.agent_runtime import AgentRuntime


def main() -> None:
    question = "안호선교수에 대해서 찾아보고, 연구자로서의 역량을 평가해줘"
    result = AgentRuntime().chat(
        question,
        "auto",
        allowed_agents=frozenset({"main", "research"}),
        allow_local_tools=False,
    )
    print(json.dumps({
        "question": question,
        "model_calls": result.llm_calls,
        "stages": result.stages,
        "tools": [
            {
                "name": tool["name"],
                "success": tool["success"],
                "duration_ms": tool["duration_ms"],
                "error": tool["error"],
                "details": tool.get("details"),
            }
            for tool in result.tools
        ],
        "agent_total_seconds": round(result.duration_ms / 1000, 3),
        "tool_seconds": round(sum(float(tool["duration_ms"]) for tool in result.tools) / 1000, 3),
        "final_visible_output_tokens": result.usage.get("completion_tokens") if result.usage else None,
        "final_answer_characters": len(result.content),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()