#!/usr/bin/env python3
"""Measure local vLLM prefill, decode, and coarse host/GPU telemetry."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass

import httpx


API_URL = os.getenv("VLLM_BENCHMARK_URL", "http://127.0.0.1:8000/v1/chat/completions")
MODEL = "qwen3.8-27b"
SAMPLES = (("500", 500, 512), ("4k", 4_000, 512), ("10k", 10_000, 512), ("decode", 100, 2_048))


@dataclass
class Telemetry:
    gpu_utilization: list[float]
    power_watts: list[float]
    graphics_clock_mhz: list[float]
    memory_clock_mhz: list[float]
    vram_mib: list[float]
    temperature_c: list[float]
    cpu_percent: list[float]
    ram_percent: list[float]
    swap_percent: list[float]


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 1) if values else None


def _telemetry_loop(stop: threading.Event, telemetry: Telemetry) -> None:
    try:
        import psutil
    except ImportError:
        return
    while not stop.wait(1):
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,power.draw,clocks.current.graphics,clocks.current.memory,memory.used,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        fields = result.stdout.strip().split(", ")
        if len(fields) == 6:
            try:
                gpu_util, power, graphics, memory, vram, temperature = (float(value) for value in fields)
                telemetry.gpu_utilization.append(gpu_util)
                telemetry.power_watts.append(power)
                telemetry.graphics_clock_mhz.append(graphics)
                telemetry.memory_clock_mhz.append(memory)
                telemetry.vram_mib.append(vram)
                telemetry.temperature_c.append(temperature)
            except ValueError:
                pass
        telemetry.cpu_percent.append(psutil.cpu_percent(interval=None))
        telemetry.ram_percent.append(psutil.virtual_memory().percent)
        telemetry.swap_percent.append(psutil.swap_memory().percent)


def _prompt(target_tokens: int) -> str:
    return ("Performance measurement context. " * target_tokens)[: target_tokens * 8]


def benchmark(name: str, target_input_tokens: int, max_output_tokens: int) -> dict[str, object]:
    telemetry = Telemetry([], [], [], [], [], [], [], [], [])
    stop = threading.Event()
    monitor = threading.Thread(target=_telemetry_loop, args=(stop, telemetry), daemon=True)
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": _prompt(target_input_tokens)}],
        "temperature": 0,
        "min_tokens": max_output_tokens,
        "max_tokens": max_output_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    started = time.perf_counter()
    first_token_at: float | None = None
    completion_tokens = 0
    usage: dict[str, int] | None = None
    monitor.start()
    try:
        with httpx.Client(timeout=900) as client, client.stream("POST", API_URL, json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                event = json.loads(line[6:])
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices", [])
                if choices and choices[0].get("delta", {}).get("content"):
                    completion_tokens += 1
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
    finally:
        stop.set()
        monitor.join(timeout=2)
    finished = time.perf_counter()
    total_seconds = finished - started
    ttft_seconds = (first_token_at - started) if first_token_at else None
    output_tokens = usage.get("completion_tokens") if usage else completion_tokens
    decode_seconds = (finished - first_token_at) if first_token_at else None
    return {
        "case": name,
        "input_tokens": usage.get("prompt_tokens") if usage else None,
        "output_tokens": output_tokens,
        "ttft_seconds": round(ttft_seconds, 3) if ttft_seconds else None,
        "total_seconds": round(total_seconds, 3),
        "decode_tokens_per_second": round(output_tokens / decode_seconds, 2) if decode_seconds and output_tokens else None,
        "end_to_end_tokens_per_second": round(output_tokens / total_seconds, 2) if output_tokens else None,
        "telemetry_mean": {
            "gpu_utilization_percent": _mean(telemetry.gpu_utilization),
            "power_watts": _mean(telemetry.power_watts),
            "graphics_clock_mhz": _mean(telemetry.graphics_clock_mhz),
            "memory_clock_mhz": _mean(telemetry.memory_clock_mhz),
            "vram_mib": _mean(telemetry.vram_mib),
            "temperature_c": _mean(telemetry.temperature_c),
            "cpu_percent": _mean(telemetry.cpu_percent),
            "ram_percent": _mean(telemetry.ram_percent),
            "swap_percent": _mean(telemetry.swap_percent),
        },
    }


def main() -> None:
    cases = SAMPLES if len(sys.argv) == 1 else tuple(sample for sample in SAMPLES if sample[0] in sys.argv[1:])
    if not cases:
        raise SystemExit(f"usage: {sys.argv[0]} [{'|'.join(sample[0] for sample in SAMPLES)}]")
    print(json.dumps([benchmark(*sample) for sample in cases], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()