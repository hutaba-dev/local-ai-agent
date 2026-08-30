#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly VENV_DIR="${VENV_DIR:-/srv/local-ai-agent/venv}"
readonly ADAPTER_HOST="${VSCODE_ADAPTER_HOST:-127.0.0.1}"
readonly ADAPTER_PORT="${VSCODE_ADAPTER_PORT:-8001}"

if [[ "${ADAPTER_HOST}" != "127.0.0.1" && "${ADAPTER_HOST}" != "::1" ]]; then
  printf 'Refusing non-local bind address: %s\n' "${ADAPTER_HOST}" >&2
  exit 2
fi

cd "${REPO_ROOT}"
exec "${VENV_DIR}/bin/python" -m uvicorn vscode_adapter.app:app \
  --host "${ADAPTER_HOST}" \
  --port "${ADAPTER_PORT}"
