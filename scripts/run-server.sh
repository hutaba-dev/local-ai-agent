#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly RUNTIME_ENV="${RUNTIME_ENV:-/etc/local-ai-agent/vllm.env}"
readonly DEFAULT_ENV="${REPO_ROOT}/infra/vllm/runtime.env.example"

set -a
source "${DEFAULT_ENV}"
if [[ -f "${RUNTIME_ENV}" ]]; then
  source "${RUNTIME_ENV}"
fi
set +a

: "${MODEL:?MODEL must be set}"
: "${SERVED_MODEL_NAME:?SERVED_MODEL_NAME must be set}"
: "${HOST:?HOST must be set}"
: "${PORT:?PORT must be set}"

if [[ "${HOST}" != "127.0.0.1" && "${HOST}" != "::1" ]]; then
  printf 'Refusing non-local bind address: %s\n' "${HOST}" >&2
  exit 2
fi

readonly VENV_DIR="${VENV_DIR:-/srv/local-ai-agent/venv}"
readonly VLLM_BIN="${VENV_DIR}/bin/vllm"

if [[ ! -x "${VLLM_BIN}" ]]; then
  printf 'vLLM is not installed at %s. Run scripts/install.sh first.\n' "${VLLM_BIN}" >&2
  exit 1
fi

mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${VLLM_CACHE_ROOT}" "${LOG_DIR}"

exec "${VLLM_BIN}" serve "${MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype "${DTYPE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --download-dir "${HF_HUB_CACHE}"
