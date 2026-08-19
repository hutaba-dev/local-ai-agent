# Qwen3.8-27B Model Serving

## Selected Serving Profile

| Setting | Value |
| --- | --- |
| Model | `Qwen/Qwen3.8-27B` |
| Server | vLLM `0.27.1` |
| GPU | 1 x NVIDIA RTX PRO 6000 Blackwell, 96 GiB VRAM |
| Precision | BF16 |
| PyTorch runtime | `2.13.0+cu129` |
| Context limit | 16,384 tokens |
| Concurrency setting | `max-num-seqs=8` |
| GPU memory target | 90% |
| CPU offload | Disabled |
| Endpoint | `http://127.0.0.1:8000/v1` |

The official Qwen3.8-27B model card identifies vLLM as a supported serving
backend. Its BF16 checkpoint is approximately 51.7 GiB. The installed RTX PRO
6000 Blackwell has 96 GiB of VRAM, so BF16 fits with room for runtime overhead
and a conservative KV cache. This is preferable to quantization for the initial
bring-up because it preserves model quality without CPU offload.

Alternative precision checkpoints are not part of the current deployment. The
Qwen FP8 and NVFP4 variants remain future evaluation candidates only; BF16 is
the validated serving profile for this RTX PRO 6000 Blackwell. MXFP4 is not
selected because it is not supported by the NVIDIA vLLM profile used here.

## Blackwell Compatibility

`flashinfer-python` incorrectly rejects the RTX PRO 6000 Blackwell sampler path
in this environment. `VLLM_USE_FLASHINFER_SAMPLER=0` disables only that sampler
path. vLLM continues using its selected supported attention backends; no NVIDIA
driver, kernel, or system CUDA toolkit change was made.

## Measured Bring-Up

The successful cold start on 2026-08-18 used the cached model files and had these
milestones:

| Milestone | Observed result |
| --- | --- |
| BF16 checkpoint size | 51.75 GiB across 18 safetensors shards |
| Model load | 5.86 seconds, 51.1 GiB GPU memory |
| vLLM engine initialize/profile/warmup | 220.38 seconds, including 77.98 seconds of `torch.compile` |
| API server ready | approximately 4 minutes 22 seconds after model loading began |
| Steady GPU allocation | approximately 86.3 GiB of 97,887 MiB |
| API validation | `/health`, `/v1/models`, and `/v1/chat/completions` passed |

The first run also downloads the checkpoint, so it can take substantially longer
than the measured cached start. Compiled artifacts are stored outside Git at
`/srv/local-ai-agent/vllm-cache`.

## Commands

```bash
./scripts/install.sh
./scripts/start-vllm.sh
./scripts/healthcheck.sh
./scripts/smoke-test.sh
```

Use SSH port forwarding for clients outside this host. Do not bind the serving
port to the public internet.