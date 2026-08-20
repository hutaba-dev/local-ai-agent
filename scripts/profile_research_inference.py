#!/usr/bin/env python3
"""Profile model calls made by one unchanged Deep Research request."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.agent_runtime import AgentRuntime


@dataclass
class CallMeasurement:
    purpose: str
    input_tokens: int | None
    output_tokens: int | None
    total_seconds: float
    output_tokens_per_second: float | None


class ProfilingClient:
    def __init__(self) -> None:
        self.client = httpx.Client(timeout=900)
        self.calls: list[CallMeasurement] = []

    def post(self, url: str, json: dict[str, object]) -> httpx.Response:
        started = time.perf_counter()
        response = self.client.post(url, json=json)
        elapsed = time.perf_counter() - started
        payload = response.json()
        usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
        messages = json.get("messages", [])
        prompt = "\n".join(
            message.get("content", "") for message in messages
            if isinstance(message, dict) and isinstance(message.get("content"), str)
        ) if isinstance(messages, list) else ""
        if "Decide whether this request" in prompt:
            purpose = "planner"
        elif isinstance(prompt, str) and "You are the Analyst / Synthesizer" in prompt:
            purpose = "analyst"
        elif isinstance(prompt, str) and "You are the Critic" in prompt:
            purpose = "critic"
        elif isinstance(prompt, str) and "Write the final research answer" in prompt:
            purpose = "final_revision"
        else:
            purpose = "other"
        output_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        self.calls.append(CallMeasurement(
            purpose=purpose,
            input_tokens=usage.get("prompt_tokens") if isinstance(usage, dict) else None,
            output_tokens=output_tokens,
            total_seconds=round(elapsed, 3),
            output_tokens_per_second=round(output_tokens / elapsed, 2) if isinstance(output_tokens, int) and elapsed else None,
        ))
        return response


def main() -> None:
    question = "인천대학교 기계공학과 안호선 교수의 연구 역량을 논문, 인용, 대표 연구와 최근 성과를 근거로 평가해줘"
    client = ProfilingClient()
    result = AgentRuntime(client=client).chat(
        question,
        "auto",
        allowed_agents=frozenset({"main", "research"}),
        allow_local_tools=False,
    )
    print(json.dumps({
        "model_calls": [asdict(call) for call in client.calls],
        "agent_total_seconds": round(result.duration_ms / 1000, 3),
        "tool_seconds": round(sum(float(tool["duration_ms"]) for tool in result.tools) / 1000, 3),
        "final_visible_output_tokens": result.usage.get("completion_tokens") if result.usage else None,
        "final_answer_characters": len(result.content),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()