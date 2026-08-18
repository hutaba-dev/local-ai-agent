#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PID_FILE="${RUN_DIR:-${REPO_ROOT}/run}/vllm.pid"

if [[ ! -f "${PID_FILE}" ]]; then
  printf 'No vLLM PID file found.\n'
  exit 0
fi

pid="$(cat "${PID_FILE}")"
if kill -0 "${pid}" 2>/dev/null; then
  kill "${pid}"
  printf 'Stopped vLLM process %s.\n' "${pid}"
else
  printf 'Removed stale vLLM PID file for process %s.\n' "${pid}"
fi
rm -f "${PID_FILE}"
