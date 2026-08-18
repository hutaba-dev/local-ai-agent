#!/usr/bin/env bash
set -euo pipefail

tail -n "${LINES:-100}" "${LOG_FILE:-/var/log/local-ai-agent/vllm.log}"
