#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly RUNTIME_ROOT="${RUNTIME_ROOT:-/srv/local-ai-agent}"
readonly VENV_DIR="${VENV_DIR:-${RUNTIME_ROOT}/venv}"
readonly REQUIREMENTS_FILE="${REPO_ROOT}/requirements/vllm-cu129.lock"

mkdir -p "${RUNTIME_ROOT}/models" "${RUNTIME_ROOT}/huggingface" \
  "${RUNTIME_ROOT}/vllm-cache" /var/log/local-ai-agent

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install \
  -r "${REQUIREMENTS_FILE}" \
  --extra-index-url https://download.pytorch.org/whl/cu129

"${VENV_DIR}/bin/python" - <<'PY'
import torch
import transformers
import vllm

assert torch.cuda.is_available(), "PyTorch cannot access CUDA"
print(f"vLLM: {vllm.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"PyTorch CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
PY

printf 'Runtime root: %s\n' "${RUNTIME_ROOT}"
printf 'Model cache: %s\n' "${RUNTIME_ROOT}/models"
printf 'Operational env file: /etc/local-ai-agent/vllm.env\n'
printf 'Repository: %s\n' "${REPO_ROOT}"
