#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
    printf 'Run this script as root on the image worker.\n' >&2
    exit 1
fi

readonly WORKER_ROOT="${IMAGE_WORKER_ROOT:-/srv/local-ai-worker}"
readonly APP_ROOT="${WORKER_ROOT}/app"
readonly TORCH_RUNTIME="${WORKER_ROOT}/torch-runtime"

test -f "${APP_ROOT}/requirements/image-worker.lock"
test -f "${WORKER_ROOT}/liveportrait/requirements.txt"
test -d "${TORCH_RUNTIME}/torch"

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg python3-venv

python3 -m venv "${WORKER_ROOT}/worker-venv"
python3 -m venv "${WORKER_ROOT}/image-venv"
python3 -m venv "${WORKER_ROOT}/pose-venv"

for venv in image-venv pose-venv; do
    site_packages="$("${WORKER_ROOT}/${venv}/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
    printf '%s\n' "${TORCH_RUNTIME}" > "${site_packages}/local-ai-torch-runtime.pth"
    "${WORKER_ROOT}/${venv}/bin/python" -m pip install \
        filelock fsspec jinja2 networkx setuptools sympy typing-extensions
done

"${WORKER_ROOT}/worker-venv/bin/python" -m pip install -r "${APP_ROOT}/requirements/image-worker.lock"

"${WORKER_ROOT}/image-venv/bin/python" -m pip install -r "${APP_ROOT}/requirements/image-backend.lock"

"${WORKER_ROOT}/pose-venv/bin/python" -m pip install -r "${WORKER_ROOT}/liveportrait/requirements.txt"
"${WORKER_ROOT}/pose-venv/bin/python" -m pip install -r "${APP_ROOT}/requirements/pose-backend.lock"

"${WORKER_ROOT}/image-venv/bin/python" - <<'PY'
import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline

assert torch.cuda.is_available(), "image backend cannot access CUDA"
print(torch.cuda.get_device_name(0), StableDiffusionPipeline.__name__, StableDiffusionImg2ImgPipeline.__name__)
PY

"${WORKER_ROOT}/pose-venv/bin/python" - <<'PY'
import cv2
import onnxruntime
import torch

assert torch.cuda.is_available(), "pose backend cannot access CUDA"
print(torch.cuda.get_device_name(0), cv2.__version__, onnxruntime.__version__)
PY