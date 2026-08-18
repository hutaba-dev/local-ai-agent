#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PID_FILE="${RUN_DIR:-${REPO_ROOT}/run}/vllm.pid"

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  printf 'vLLM process is running: PID %s\n' "$(cat "${PID_FILE}")"
  "${REPO_ROOT}/scripts/healthcheck.sh"
else
  printf 'vLLM process is not running.\n'
  exit 1
fi
