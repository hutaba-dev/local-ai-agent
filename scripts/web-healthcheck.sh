#!/usr/bin/env bash
set -euo pipefail

readonly WEB_HEALTH_URL="${WEB_HEALTH_URL:-http://127.0.0.1:8080/health}"

curl --fail --silent --show-error --max-time 10 "${WEB_HEALTH_URL}" \
  | grep -q '"status":"ok"'
printf 'Web UI health check passed: %s\n' "${WEB_HEALTH_URL}"