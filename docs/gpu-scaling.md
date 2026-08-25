# Dual-GPU vLLM Scaling Investigation

Validated on 2026-08-25 for the Qwen3.8-27B service.

## Status

Production runs TP2 with direct GPU peer communication disabled. The validated
configuration uses `NCCL_P2P_DISABLE=1` and vLLM's
`--disable-custom-all-reduce`, which selects PYNCCL without triggering the AMD-Vi
fault. Direct-P2P TP2 reproducibly triggers AMD-Vi `IO_PAGE_FAULT` events from
GPU 1 during NCCL initialization and must not be enabled on this platform.

## Hardware

| GPU | Model | VRAM | PCI bus | Idle PCIe link | UUID |
| --- | --- | ---: | --- | --- | --- |
| GPU 0 | NVIDIA RTX PRO 6000 Blackwell Workstation Edition | 97,887 MiB | `00000000:21:00.0` | Gen1 x8 (maximum Gen4 x16) | `GPU-9a329ab4-f223-e556-f2f7-7e9e044dcd11` |
| GPU 1 | NVIDIA RTX PRO 6000 Blackwell Workstation Edition | 97,887 MiB | `00000000:4B:00.0` | Gen1 x16 (maximum Gen4 x16) | `GPU-2218849a-a940-80ec-2300-99f56cac511f` |

Both GPUs are attached to NUMA node 0. `nvidia-smi topo -m` reports `NODE` between GPU 0 and GPU 1. There is no NVLink path; tensor-parallel communication traverses PCIe host bridges within the same NUMA node. Idle PCIe generation is expected to downshift. No volatile uncorrectable ECC error was reported.

```text
        GPU0    GPU1    CPU Affinity    NUMA Affinity
GPU0     X      NODE    0-127           0
GPU1    NODE     X      0-127           0
```

## Runtime Configuration

vLLM version: `0.27.1`

The API contract remains unchanged:

- Model: `Qwen/Qwen3.8-27B`
- Served model name: `qwen3.8-27b`
- Bind: `127.0.0.1:8000`
- dtype: `bfloat16`
- Quantization: none
- GPU memory utilization: `0.90`
- Maximum model length: `16,384`
- Maximum sequences: `8`
- Reasoning parser: `qwen3`
- Tool-call parser: `qwen3_coder`
- Automatic tool choice: enabled
- KV cache dtype: automatic/default
- CPU offload: disabled/default
- Swap space: vLLM default
- Pipeline parallel size: `1`

The active pre-change service used a 16,384-token maximum model length, despite older architecture notes describing 128K. The dual-GPU change preserves the verified active value instead of introducing a context-size change at the same time.

### One GPU Baseline

- `CUDA_VISIBLE_DEVICES`: effectively GPU 0
- Tensor parallel size: `1`
- Model memory: 51.1 GiB
- Available KV cache: 31.8 GiB
- Idle VRAM: GPU 0 approximately 86,327 MiB; GPU 1 approximately 18 MiB

### Active Two-GPU Configuration

- `CUDA_VISIBLE_DEVICES=0,1`
- `TENSOR_PARALLEL_SIZE=2`
- vLLM argument: `--tensor-parallel-size 2`
- `NCCL_P2P_DISABLE=1`
- vLLM argument: `--disable-custom-all-reduce`
- Collective backend: PYNCCL
- Pipeline parallel size remains `1`

The settings are stored in `/etc/local-ai-agent/vllm.env`. Removing that file
restores the script's TP1 defaults. The image generation and LivePortrait
service architecture is unchanged.

## Benchmark Method

All cases use the same OpenAI-compatible Chat Completions endpoint, deterministic temperature, thinking disabled, streaming responses, and forced length completion. GPU utilization, VRAM, and power are sampled approximately every 200 ms with `nvidia-smi`.

| Case | Input target | Output target |
| --- | ---: | ---: |
| Short | 500 tokens | 512 tokens |
| Medium | 2,000 tokens | 1,024 tokens |
| Long | 10,000 tokens | 2,048 tokens |

TTFT includes HTTP request, prompt processing, and the first streamed content token. Decode throughput is output tokens divided by elapsed time from first content token to stream completion.

## Benchmark Results

### TP1 Baseline

| Case | Actual input | Output | TTFT | Decode | Total | GPU 0 utilization avg/max | GPU 0 VRAM max | GPU 0 power avg/max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Short | 516 | 512 | 0.310 s | 29.68 tok/s | 17.559 s | 95.4% / 100% | 86,327 MiB | 526.0 / 562.2 W |
| Medium | 2,028 | 1,024 | 0.367 s | 29.53 tok/s | 35.040 s | 96.7% / 100% | 86,599 MiB | 572.0 / 587.0 W |
| Long | 10,020 | 2,048 | 1.785 s | 29.20 tok/s | 71.914 s | 99.6% / 100% | 88,567 MiB | 592.2 / 601.4 W |

GPU 1 remained at 0% utilization and approximately 18 MiB during all TP1 cases.

### Direct-P2P TP2 Failure

No valid TP2 benchmark was collected. Both NCCL ranks were created, but GPU 1
reported four IOMMU DMA faults before model loading completed:

```text
nvidia 0000:4b:00.0: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0025 address=0xac826200 flags=0x0020]
nvidia 0000:4b:00.0: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0025 address=0x90301000 flags=0x0020]
nvidia 0000:4b:00.0: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0025 address=0xac826200 flags=0x0020]
nvidia 0000:4b:00.0: AMD-Vi: Event logged [IO_PAGE_FAULT domain=0x0025 address=0x90300000 flags=0x0020]
```

The same four faults at the same addresses were reproduced after the abrupt
host reset when systemd retried TP2 at boot. The first boot ended without a
normal shutdown sequence; `last -x` marks active sessions as `crash`. No OOM,
NVIDIA Xid, PCIe AER error, ECC error, or kernel panic was recorded.

### P2P-Disabled TP2

| Case | Actual input | Output | TP1 decode | TP2 decode | Speedup | TP2 TTFT | TP2 total | Mean GPU utilization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500 | 497 | 512 | 29.66 tok/s | 51.00 tok/s | 1.719x | 0.947 s | 10.986 s | 90.1% |
| 4K | 3,890 | 512 | 29.48 tok/s | 50.69 tok/s | 1.719x | 1.460 s | 11.560 s | 99.1% |
| 10K | 9,710 | 512 | 29.28 tok/s | 50.49 tok/s | 1.724x | 3.344 s | 13.485 s | 94.7% |
| Decode | 110 | 2,048 | 29.55 tok/s | 50.82 tok/s | 1.720x | 0.778 s | 41.079 s | 97.2% |

Compared with the matching TP1 cases, decode speedup is 1.719-1.724x and
two-GPU scaling efficiency is 86.0-86.2%. No IOMMU, Xid, AER, OOM, or kernel
panic event was recorded during startup, API smoke tests, or the benchmark.

First-time TP2 startup took approximately 20 minutes: engine initialization was
911.47 seconds, including 440.71 seconds of Torch compilation and 379.08 seconds
of initial profiling/warmup, followed by 164.67 seconds of multimodal warmup.
On the validated persistent restart, the generated AOT cache reduced engine
initialization to 44.69 seconds, compilation to 7.54 seconds, and initial
profiling/warmup to 4.78 seconds. The service restart completed in 9 minutes 53
seconds, including model/tokenizer setup and 135.08 seconds of multimodal
warmup.

The unchanged Deep Research request completed all four model calls at
50.05-51.28 tok/s. Its 29.25-second total is not directly comparable with the
earlier 221.37-second run because no web-search credentials were available and
the search tool failed immediately.

The production integration check also passed with TP2 active: Web health was
OK, SD-Turbo generated a 512-pixel PNG, LivePortrait processed that image, and
Qwen returned a valid chat completion after both media models remained loaded.
All five systemd services remained active.

## Scaling

For each workload:

```text
speedup = TP2 decode tokens/s / TP1 decode tokens/s
scaling efficiency = speedup / 2
```

Scaling efficiency is diagnostic rather than a pass/fail threshold. This system has a PCIe `NODE` topology without NVLink, so collective communication overhead is expected.

## Rollback

The verified TP1 files are backed up outside the repository at:

```text
/root/local-ai-agent-backups/qwen-vllm-tp1-20260825/
```

Rollback procedure:

1. Remove `/etc/local-ai-agent/vllm.env`, or remove its three TP2 variables.
2. Run `systemctl unset-environment TENSOR_PARALLEL_SIZE DISABLE_CUSTOM_ALL_REDUCE NCCL_P2P_DISABLE`.
3. Restart `qwen-vllm.service`.
4. Confirm `/v1/models` and `/v1/chat/completions`.
5. Confirm only GPU 0 contains the vLLM engine.
6. To restore local image features, remove both `llm-dedicated.conf` drop-ins, then run `systemctl daemon-reload` and `systemctl enable --now local-ai-image.service local-ai-pose.service`.

Do not expose port 8000 or change the localhost bind during rollback.

## Known Issues

- GPU-to-GPU topology is PCIe `NODE`, not NVLink.
- GPU 1 triggers reproducible AMD-Vi IOMMU page faults when NCCL initializes
        direct-P2P TP2. Keep `NCCL_P2P_DISABLE=1` and custom all-reduce disabled.
- The Gigabyte TRX40 AORUS XTREME BIOS is version `FA` dated 2020-08-04. Firmware compatibility should be investigated before another TP2 attempt; no BIOS, kernel, driver, or IOMMU setting was changed during this work.
- The display server uses a small amount of memory on both GPUs.
- GPU 0 reports an idle x8 link while its maximum is x16; link state under load should be considered when investigating unexpectedly weak scaling.
- Transformer video processor documentation warnings for `min_frames` and `max_frames` are emitted as errors but were also present in the working TP1 startup and are non-fatal.
