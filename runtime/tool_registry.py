"""Read-only whitelist tools for the initial browser agent evaluation."""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter


REPO_ROOT = Path(__file__).resolve().parents[1]
MAX_OUTPUT_CHARS = 12_000


@dataclass(frozen=True)
class ToolResult:
    name: str
    success: bool
    output: str
    error: str | None
    duration_ms: int


def run_agent_tools(agent: str, message: str) -> list[dict[str, object]]:
    tools = {
        "coding": _coding_tools,
        "research": _research_tools,
        "server": _server_tools,
    }.get(agent, lambda _: [])
    return [asdict(result) for result in tools(message)]


def _coding_tools(message: str) -> list[ToolResult]:
    return [
        _command("list_files", ["find", ".", "-maxdepth", "2", "-type", "f", "-not", "-path", "./.git/*"], cwd=REPO_ROOT),
        _command("search_files", ["git", "grep", "-n", "Qwen3.8-27B", "--", "README.md", "docs"], cwd=REPO_ROOT),
        _command("read_file", ["sed", "-n", "1,220p", "README.md"], cwd=REPO_ROOT),
        _command("git_status", ["git", "status", "--short", "--branch"], cwd=REPO_ROOT),
        _command("git_diff", ["git", "diff", "--stat"], cwd=REPO_ROOT),
    ]


def _research_tools(message: str) -> list[ToolResult]:
    return [
        _command("search_project_docs", ["find", "docs", "-type", "f", "-name", "*.md", "-print"], cwd=REPO_ROOT),
        _command("read_file", ["sed", "-n", "1,220p", "docs/model-serving.md"], cwd=REPO_ROOT),
    ]


def _server_tools(message: str) -> list[ToolResult]:
    return [
        _command("nvidia_smi", ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"]),
        _command("systemctl_status_qwen_vllm", ["systemctl", "is-active", "qwen-vllm.service"]),
        _command("journalctl_qwen_vllm", ["journalctl", "-u", "qwen-vllm.service", "-n", "12", "--no-pager"]),
        _command("df", ["df", "-h", "/"]),
        _command("free", ["free", "-h"]),
    ]


def _command(name: str, command: list[str], cwd: Path | None = None) -> ToolResult:
    started = perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        output = completed.stdout[-MAX_OUTPUT_CHARS:].strip()
        error = completed.stderr[-MAX_OUTPUT_CHARS:].strip() or None
        return ToolResult(name, completed.returncode == 0, output, error, round((perf_counter() - started) * 1000))
    except (OSError, subprocess.TimeoutExpired) as error:
        return ToolResult(name, False, "", str(error), round((perf_counter() - started) * 1000))