#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly RUN_DIR="${RUN_DIR:-${REPO_ROOT}/run}"
readonly PID_FILE="${RUN_DIR}/vllm.pid"
readonly LOG_FILE="${LOG_FILE:-/var/log/local-ai-agent/vllm.log}"

mkdir -p "${RUN_DIR}" "$(dirname "${LOG_FILE}")"

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  printf 'vLLM is already running with PID %s\n' "$(cat "${PID_FILE}")" >&2
  exit 1
fi

rm -f "${PID_FILE}"
nohup "${REPO_ROOT}/scripts/run-server.sh" >>"${LOG_FILE}" 2>&1 &
printf '%s\n' "$!" > "${PID_FILE}"
printf 'Started vLLM with PID %s. Logs: %s\n' "$!" "${LOG_FILE}"