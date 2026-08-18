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

## Future Lower-VRAM Profile

For a single 48 GiB card, do not use BF16. The supported alternatives are:

| Checkpoint | vLLM support | Approximate checkpoint footprint | Trade-off |
| --- | --- | --- | --- |
| `Qwen/Qwen3.8-27B-FP8` | Native FP8 on Blackwell | 38 GiB | Better quality margin than 4-bit, limited KV room on 48 GiB |
| `Inferact/Qwen3.8-27B-NVFP4` | Native NVFP4 on Blackwell | 32 GiB | Fits 48 GiB with more KV cache; quantization quality trade-off |

The recipe explicitly warns that MXFP4 is not supported for NVIDIA vLLM serving. Quantized profiles should be tested independently before replacing the BF16 profile.
