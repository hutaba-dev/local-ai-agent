# Model And Serving Decision

## Selected Initial Profile

- Model: `Qwen/Qwen3.8-27B`
- Serving backend: vLLM `0.27.1`
- Python: isolated Python 3.12 virtual environment
- Precision: BF16
- GPU topology: one RTX PRO 6000 Blackwell (96 GiB)
- Context cap: 16,384 tokens for the first production validation
- API binding: `127.0.0.1:8000` only

The official Qwen model card identifies the model as a 27B multimodal dense model and recommends vLLM for serving. The official vLLM recipe requires vLLM `0.17.0+` and `transformers >= 5.8.0`; this repository pins vLLM `0.27.1`, the current installable stable release when this decision was made.

The official recipe estimates BF16 weights at 51.7 GiB. The installed 96 GiB GPU leaves enough capacity for BF16 weights, runtime overhead, and a deliberately capped KV cache. CPU offload is not used.

## Alternative Precision Profiles

The current server uses BF16 and does not need a quantized checkpoint. If a
future evaluation requires a different precision, test it independently against
the RTX PRO 6000 Blackwell baseline before replacing the BF16 profile. The Qwen
checkpoints `Qwen/Qwen3.8-27B-FP8` and `Inferact/Qwen3.8-27B-NVFP4` are possible
evaluation candidates. MXFP4 is not supported for the NVIDIA vLLM serving
profile used here.
