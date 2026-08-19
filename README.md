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

For VS Code Stable Custom Endpoint values, SSH tunnel access, and connection
prompts, see [docs/vscode-local-model.md](docs/vscode-local-model.md).

For a remote VS Code session on this host, configure the client to use that base URL. For a client on another machine, use an SSH port forward instead of exposing port 8000 publicly:

```bash
ssh -L 8000:127.0.0.1:8000 root@KIM_SERVER
```

## systemd Service

[infra/systemd/qwen-vllm.service](infra/systemd/qwen-vllm.service) is the
systemd template for this host. It runs as `root` because the verified model,
cache, and virtual environment paths are currently owned by `root`. It invokes
`/root/local-ai-agent/scripts/run-server.sh`, which uses
`/srv/local-ai-agent/venv/bin/vllm`.

Keep operational overrides and any secrets in `/etc/local-ai-agent/vllm.env`.
The unit does not embed secrets. Install the reviewed template, then manage it
with these commands:

```bash
sudo install -m 0644 infra/systemd/qwen-vllm.service /etc/systemd/system/qwen-vllm.service
sudo systemctl daemon-reload
sudo systemctl enable --now qwen-vllm.service
sudo systemctl start qwen-vllm.service
sudo systemctl stop qwen-vllm.service
sudo systemctl restart qwen-vllm.service
sudo systemctl status qwen-vllm.service
sudo journalctl -u qwen-vllm.service -f
```

The service sends stdout and stderr to journald. For recent logs, use
`sudo journalctl -u qwen-vllm.service -n 200 --no-pager`.

### Reboot Validation

After enabling the unit, validate reboot persistence during a maintenance
window:

```bash
sudo reboot
# After reconnecting:
sudo systemctl is-enabled qwen-vllm.service
sudo systemctl status qwen-vllm.service
./scripts/healthcheck.sh
./scripts/smoke-test.sh
sudo journalctl -u qwen-vllm.service -b --no-pager
```

`is-enabled` should report `enabled`, and both API checks should pass. The first
post-reboot start can take several minutes while vLLM initializes the engine.

## Documentation

- [docs/server-diagnosis.md](docs/server-diagnosis.md)
- [docs/model-decision.md](docs/model-decision.md)
- [docs/environment-baseline.md](docs/environment-baseline.md)
- [docs/decisions.md](docs/decisions.md)
- [docs/model-serving.md](docs/model-serving.md)
- [docs/agent-evaluation.md](docs/agent-evaluation.md)
- [docs/architecture.md](docs/architecture.md)
