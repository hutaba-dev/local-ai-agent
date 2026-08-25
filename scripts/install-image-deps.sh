#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly VENV_DIR="${VENV_DIR:-/srv/local-ai-agent/venv}"

"${VENV_DIR}/bin/python" -m pip install -r "${REPO_ROOT}/requirements/image.lock"
"${VENV_DIR}/bin/python" - <<'PY'
import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline

assert torch.cuda.is_available(), "PyTorch cannot access CUDA"
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Pipelines: {StableDiffusionPipeline.__name__}, {StableDiffusionImg2ImgPipeline.__name__}")
PY