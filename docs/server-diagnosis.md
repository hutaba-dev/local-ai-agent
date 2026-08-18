# Server Diagnosis

Collected on 2026-08-18 before installing the local AI service.

| Component | Result |
| --- | --- |
| OS | Ubuntu 24.04.2 LTS |
| Kernel | Linux 7.0.0-28-generic |
| CPU | AMD Ryzen Threadripper 3990X, 64 cores / 128 threads |
| RAM | 251 GiB total, 245 GiB available during diagnosis |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB VRAM |
| NVIDIA driver | 580.142 |
| Driver CUDA capability | CUDA 13.0 |
| CUDA toolkit | 12.8.93 (`nvcc`) |
| Python | 3.12.3 |
| Docker | 29.1.3 |
| Git | 2.43.0 |
| Root disk | 937 GiB total, 691 GiB available |

## Findings

- The GPU was idle except for Xorg and GNOME Shell using 111 MiB combined.
- Docker has `nvidia-container-runtime` installed, but `docker run --gpus all` fails because the daemon cannot select a GPU device driver. This deployment uses an isolated Python virtual environment and does not alter Docker, NVIDIA drivers, CUDA, the kernel, or partitions.
- `gh` is not installed. GitHub push remains pending until a credential-safe remote is configured; no token is stored by this repository.
