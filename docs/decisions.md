# Architecture Decisions

## Model And Hardware

- The baseline model is `Qwen/Qwen3.8-27B` served in BF16 with vLLM.
- The actual installed accelerator is **one NVIDIA RTX PRO 6000 Blackwell with
  96 GiB VRAM**, not an NVIDIA A6000.
- Start with one RTX PRO 6000 Blackwell. Future scale-out to two matching RTX
  PRO 6000 Blackwell GPUs must be validated with tensor-parallel serving before
  becoming the default profile.

## Service Boundaries

- The serving API is a dedicated vLLM process with an OpenAI-compatible
  endpoint at `http://127.0.0.1:8000/v1`.
- Agent orchestration and any UI are separate clients of the serving API. They
  must not be coupled to the model server lifecycle or implementation details.
- The serving API binds to localhost only. Remote access is provided through an
  SSH port forward, never by opening the inference port directly.

## Source Of Truth

- This Git repository and its GitHub remote are the source of truth for
  infrastructure, scripts, documentation, and non-secret configuration.
- Runtime state, model files, logs, local agent memory, credentials, and tokens
  remain outside Git.