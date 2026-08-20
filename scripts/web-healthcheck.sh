#!/usr/bin/env bash
set -euo pipefail

readonly WEB_HEALTH_URL="${WEB_HEALTH_URL:-http://127.0.0.1:8080/health}"
readonly ATTEMPTS="${WEB_HEALTH_ATTEMPTS:-10}"

for attempt in $(seq 1 "${ATTEMPTS}"); do
  if curl --fail --silent --show-error --max-time 2 "${WEB_HEALTH_URL}" \
    | grep -q '"status":"ok"'; then
    printf 'Web UI health check passed: %s\n' "${WEB_HEALTH_URL}"
    exit 0
  fi
  if [[ "${attempt}" -lt "${ATTEMPTS}" ]]; then
    sleep 1
  fi
done

printf 'Web UI health check failed after %s attempts: %s\n' "${ATTEMPTS}" "${WEB_HEALTH_URL}" >&2
exit 1