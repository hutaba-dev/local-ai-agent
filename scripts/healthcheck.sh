#!/usr/bin/env bash
set -euo pipefail

readonly BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"

curl --fail --silent --show-error --max-time 10 "${BASE_URL}/health" >/dev/null
curl --fail --silent --show-error --max-time 10 "${BASE_URL}/v1/models" \
  | grep -q '"id"'
printf 'vLLM health check passed: %s\n' "${BASE_URL}"
