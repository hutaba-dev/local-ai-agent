# Environment Baseline

Collected on 2026-08-19. No system packages were upgraded while collecting this
baseline.

| Component | Observed value |
| --- | --- |
| OS | Ubuntu 24.04.2 LTS |
| Kernel | `7.0.0-28-generic` |
| CPU | AMD Ryzen Threadripper 3990X, 64 cores / 128 threads |
| Memory | 251 GiB total, 238 GiB available |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| GPU VRAM | 97,887 MiB (approximately 96 GiB) |
| NVIDIA driver | 580.142 |
| Driver CUDA runtime capability | CUDA 13.0 |
| Installed CUDA toolkit | CUDA 12.8.93 (`nvcc`) |
| Python | 3.12.3 |
| pip | 24.0 (system Python) |
| uv | Not installed |
| venv | Available through Python 3.12 |
| Docker | 29.1.3 |
| Git | 2.43.0 |
| Root filesystem | 937 GiB total, 625 GiB available |

## Runtime State

The local vLLM deployment uses an isolated virtual environment at
`/srv/local-ai-agent/venv`. The active Qwen3.8-27B BF16 server runs on the RTX
PRO 6000 Blackwell GPU, binds only to `127.0.0.1:8000`, and exposes an
OpenAI-compatible API.

At collection time, `VLLM::EngineCore` was the expected compute process and
occupied approximately 86 GiB of GPU memory. Xorg and GNOME Shell also use a
small amount of display memory.

## Constraints

- Do not expose the serving port publicly; use SSH forwarding for remote users.
- Do not commit model weights, Hugging Face caches, runtime logs, local memory
  databases, virtual environments, or secrets.
- Docker is installed, but GPU access through Docker has not been validated for
  this deployment. The supported runtime is the isolated Python vLLM install.