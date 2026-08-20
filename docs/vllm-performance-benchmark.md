# Qwen3.8-27B Inference Performance Benchmark

## Scope and Conclusion

This investigation keeps the Research Agent v3 architecture, research loop,
tools, and prompts unchanged. It measures the serving path behind the reported
approximately `10.5 tok/s` result.

The model's measured single-request decode rate is approximately `29.3-29.7
tok/s`, not `10.5 tok/s`. The lower user-facing number is an aggregate metric:
it divides the elapsed time of four sequential model calls in the Research v3
workflow by the final visible answer token count. It is not evidence of CPU
offload, swapping, a fallback attention backend, or a low GPU clock.

No serving configuration was changed in production. The only one-variable
optimization experiment did not become API-ready, so it has no valid after
measurement and is not deployed.

## Hardware

| Item | Value |
| --- | --- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition |
| GPU VRAM | 97,887 MiB (approximately 95.6 GiB) |
| Compute capability | SM120 / `(12, 0)` |
| Driver | 580.142 |
| NVIDIA-SMI CUDA version | 13.0 |
| Power limit | 600 W |
| Host RAM | 251 GiB total; approximately 237 GiB available at baseline |
| Host swap | 8 GiB total; 0 used during tests |

## Software Stack

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen3.8-27B` (`Qwen3_5ForConditionalGeneration`) |
| vLLM | `0.27.1`, V1 engine |
| PyTorch | `2.13.0+cu129` |
| PyTorch CUDA runtime | 12.9 |
| Precision | BF16 (`torch.bfloat16`) |
| Quantization | None |
| Attention | FlashAttention 2 |
| Qwen GDN prefill | Triton/FLA kernel |
| Execution | CUDA graphs and `torch.compile`/Inductor enabled |
| Tensor/pipeline/data parallelism | 1 / 1 / 1 |

FlashInfer `0.6.16.post3` is installed but reports `SM 12.x requires CUDA >=
12.9` while probing capability. The sampler is therefore intentionally kept
disabled with `VLLM_USE_FLASHINFER_SAMPLER=0`. This does not disable the active
FlashAttention 2 attention backend.

## Current Serving Configuration

The `qwen-vllm.service` unit launches [scripts/run-server.sh](../scripts/run-server.sh),
which serves on `127.0.0.1:8000` with:

```text
--dtype bfloat16
--gpu-memory-utilization 0.90
--max-model-len 16384
--max-num-seqs 8
--reasoning-parser qwen3
--enable-auto-tool-choice
--tool-call-parser qwen3_coder
```

There is no `--cpu-offload-gb`, no engine CPU offload configuration, no process
swap (`VmSwap: 0 kB`), and no system swap use. Startup profiling measured 51.67
GiB for weights/non-torch allocations, 2.0 GiB peak activation, 0.1 GiB CUDA
graphs, and 31.8 GiB available KV cache. The KV cache holds 453,290 tokens,
equivalent to 27.67 concurrent 16,384-token requests by vLLM's estimate.

## Benchmark Method

[scripts/vllm_performance_benchmark.py](../scripts/vllm_performance_benchmark.py)
uses a single streaming OpenAI-compatible request with `temperature=0`, fixed
output length, and `enable_thinking=false`. It records API usage tokens, time to
first token (TTFT), decode throughput, end-to-end throughput, and one-second
GPU/host telemetry samples. The measurements below use the production server.

## Baseline: Prefill and Decode

| Case | Input tokens | Output tokens | TTFT | Decode tok/s | End-to-end tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 500-token prompt | 497 | 512 | 0.166 s | 29.66 | 29.38 |
| 4K-token prompt | 3,890 | 512 | 0.716 s | 29.48 | 28.31 |
| 10K-token prompt | 9,710 | 512 | 1.734 s | 29.28 | 26.64 |
| Decode-focused | 110 | 2,048 | 0.119 s | 29.55 | 29.50 |

The 10K request has a modest prefill cost: TTFT increases by 1.568 seconds from
the 500-token case, while decode remains at 29.28 tok/s. The configured 16K
context limit is therefore not causing the approximately 200-second Research
request duration.

## GPU Utilization Analysis

| Case | GPU util. | Power | Graphics clock | Memory clock | VRAM | CPU | RAM | Swap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 500-token prompt | 100% | 554.8 W | 2815.5 MHz | 13365 MHz | 91037 MiB | 0.9% | 5.7% | 0% |
| 4K-token prompt | 100% | 578.4 W | 2787.8 MHz | 13365 MHz | 91037 MiB | 1.0% | 5.7% | 0% |
| 10K-token prompt | 100% | 593.8 W | 2716.6 MHz | 13365 MHz | 91037 MiB | 0.8% | 5.7% | 0% |
| Decode-focused | 99.3% | 599.6 W | 2772.9 MHz | 13365 MHz | 91037 MiB | 0.9% | 5.7% | 0% |

The GPU is saturated and near its 600 W power limit during generation. CPU use,
RAM use, and swap use are negligible. Together with the selected FlashAttention
2 and Triton/FLA kernels, this rules out a host-memory or generic-backend
bottleneck for the observed single-request throughput.

## Research v3 Call Profile

[scripts/profile_research_inference.py](../scripts/profile_research_inference.py)
ran the unchanged Korean researcher evaluation request through `AgentRuntime`.
It instruments the HTTP client only; it does not change the agent behavior.

| Call | Input tokens | Output tokens | Total | Output tok/s |
| --- | ---: | ---: | ---: | ---: |
| Planner | 227 | 137 | 4.689 s | 29.21 |
| Analyst / Synthesizer | 5,144 | 2,443 | 84.827 s | 28.80 |
| Critic | 7,547 | 1,150 | 40.958 s | 28.08 |
| Final revision | 8,725 | 2,217 | 78.122 s | 28.38 |

The complete agent request took 221.370 seconds, including 12.772 seconds in
tools. The final visible answer contained 2,217 tokens. Dividing the full
workflow time by only those 2,217 final tokens yields about 10 tok/s, although
every individual model call decoded at about 28-29 tok/s. This is the source of
the reported approximately `10.5 tok/s` figure.

`academic_papers` failed in approximately 535 ms during the request. That is a
separate reliability issue and is not a material contributor to inference
latency.

## Optimization Experiment

The only safe candidate was `--max-num-batched-tokens 16384`, evaluated because
the baseline uses vLLM's default `8192` token batch budget while the benchmark
contains a 9,710-token prompt. Model, BF16 precision, context limit, VRAM
target, sampling path, and all agent behavior were held constant.

| Configuration | Scheduler/compile range | Result | Production status |
| --- | --- | --- | --- |
| Baseline | 8192 / `[8192]` | 10K TTFT 1.734 s; decode 29.28 tok/s | Active |
| Candidate | 16384 / `[16384]` | Engine loaded weights but did not expose an API listener after encoder profiling/compile; no benchmark result | Not deployed |

The candidate was stopped and the baseline service was restored. Because it
never produced an API response, there is no valid before/after throughput
comparison. Claiming an improvement would be unsupported.

## Recommended Production Configuration

Keep the current validated production configuration unchanged:

- BF16 with no quantization, no CPU offload, `gpu-memory-utilization=0.90`.
- FlashAttention 2 and Triton/FLA GDN prefill selected automatically by vLLM.
- `VLLM_USE_FLASHINFER_SAMPLER=0` until a Blackwell/SM120-compatible FlashInfer
  package has been validated in a controlled test.
- `max-num-batched-tokens=8192`; do not deploy the 16384 candidate without a
  completed readiness and API benchmark.

Increasing displayed Research throughput without modifying the pipeline would
require a different metric: report model-call decode throughput separately from
whole-workflow elapsed time. This report intentionally does not collapse,
parallelize, or otherwise alter the Research v3 sequence.