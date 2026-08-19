#!/usr/bin/env bash
set -euo pipefail

readonly BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
readonly MODEL="${SERVED_MODEL_NAME:-qwen3.8-27b}"

"$(dirname "${BASH_SOURCE[0]}")/healthcheck.sh"

models_json="$(curl --fail --silent --show-error "${BASE_URL}/v1/models")"
printf '%s' "${models_json}" | python3 -c '
import json
import sys

model = sys.argv[1]
payload = json.load(sys.stdin)
assert any(item.get("id") == model for item in payload["data"])
' "${MODEL}"

response="$(curl --fail --silent --show-error --max-time 180 \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with exactly: local agent ready\"}],\"max_tokens\":32,\"temperature\":0}" \
  "${BASE_URL}/v1/chat/completions")"

printf '%s' "${response}" | python3 -c '
import json
import sys

payload = json.load(sys.stdin)
assert payload["choices"], "response has no choices"
message = payload["choices"][0]["message"]
assert message.get("content") or message.get("reasoning"), "response has no generated assistant text"
'
printf 'Chat completion smoke test passed: %s\n' "${BASE_URL}"
