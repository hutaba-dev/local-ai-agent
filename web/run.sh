#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly VENV_DIR="${VENV_DIR:-/srv/local-ai-agent/venv}"
readonly WEB_HOST="${WEB_HOST:-0.0.0.0}"
readonly WEB_PORT="${WEB_PORT:-7000}"

cd "${REPO_ROOT}"
if [[ -f .env ]]; then
	set -a
	source .env
	set +a
fi
exec "${VENV_DIR}/bin/python" -m uvicorn web.app:app --host "${WEB_HOST}" --port "${WEB_PORT}"