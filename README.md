# Local AI Agent Server

Private Qwen3.8-27B OpenAI-compatible server for local VS Code use. The service binds only to `127.0.0.1:8000`.

This repository is the source of truth for non-secret deployment configuration,
operations scripts, and architecture decisions. The baseline server has one
NVIDIA RTX PRO 6000 Blackwell GPU with 96 GiB VRAM.

## Install

```bash
cd /root/local-ai-agent
./scripts/install.sh
```

The virtual environment is `/srv/local-ai-agent/venv`. Hugging Face model files are downloaded into `/srv/local-ai-agent/models`; other Hugging Face state is under `/srv/local-ai-agent/huggingface`. None of these paths are tracked by Git.

Optional operational overrides belong in `/etc/local-ai-agent/vllm.env`, based on [infra/vllm/runtime.env.example](infra/vllm/runtime.env.example). Do not place tokens or other secrets in the repository.

## Operations

```bash
./scripts/start-vllm.sh
./scripts/status.sh
./scripts/logs.sh
./scripts/stop.sh
./scripts/healthcheck.sh
./scripts/smoke-test.sh
```

The initial model download is large and occurs during the first `start.sh`. Do not interrupt it unless the log reports a failure.

## API And VS Code

Use the OpenAI-compatible base URL `http://127.0.0.1:8000/v1` and model name `qwen3.8-27b`. The smoke test validates both `GET /v1/models` and `POST /v1/chat/completions`.

For a remote VS Code session on this host, configure the client to use that base URL. For a client on another machine, use an SSH port forward instead of exposing port 8000 publicly:

```bash
ssh -L 8000:127.0.0.1:8000 root@KIM_SERVER
```

## systemd Draft

[infra/systemd/local-ai-agent-vllm.service](infra/systemd/local-ai-agent-vllm.service) is a draft only. It is intentionally not copied into `/etc/systemd/system` and not enabled. Review the runtime environment file and service user before enabling it.

## Documentation

- [docs/server-diagnosis.md](docs/server-diagnosis.md)
- [docs/model-decision.md](docs/model-decision.md)
- [docs/environment-baseline.md](docs/environment-baseline.md)
- [docs/decisions.md](docs/decisions.md)
- [docs/model-serving.md](docs/model-serving.md)
