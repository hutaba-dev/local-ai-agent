# Distributed Image Worker

Persistent Projects, conversations, files, artifacts, and long-term memory are owned by KIM. See [projects-and-memory.md](projects-and-memory.md). AHN7 remains an image-only compute worker and does not retain Project state.

## Architecture

The Main Server remains the only user-facing host. Web and Telegram call
`runtime/image_client.py`, which routes these capabilities to AHN7:

- `image.generate`
- `image.edit`
- `portrait.frontalize`

The worker binds only to AHN7 loopback at `127.0.0.1:8010`. Main reaches it
through `local-ai-worker-tunnel.service` at `127.0.0.1:18010`. The same SSH
connection reverse-forwards Main vLLM to AHN7 loopback `127.0.0.1:18000`, so
SD-Turbo prompt optimization is preserved without exposing either API.

Requests require a bearer token stored only in root-owned environment files:

- Main: `/etc/local-ai-agent/image-worker.env`
- AHN7: `/etc/local-ai-worker/worker.env`

Do not commit either file or print its token.

## GPU Arbitration

The worker accepts one GPU task at a time. It runs only one backend process:
SD-Turbo or LivePortrait. Switching capabilities terminates the current
backend before starting the other, preventing simultaneous model residency on
the 8 GiB RTX 5060. There is no automatic fallback to either Main GPU.

Validated production measurements on RTX 5060:

| Operation | Latency | Resident VRAM |
| --- | ---: | ---: |
| SD-Turbo generate, cold | 5.48 s | 3919 MiB |
| SD-Turbo edit, warm | 1.29 s | 3919 MiB |
| LivePortrait frontalize, cold | 39.90 s | 1499 MiB |

## Deployment

AHN7 uses `/srv/local-ai-worker`. The source and model cache are copied from
the validated Main deployment. LivePortrait revision is
`9b294b3d0536135442ea73cb01e6cb3ca7029dd3`.

```bash
# AHN7
bash /srv/local-ai-worker/app/scripts/install-image-worker.sh
systemctl enable --now local-ai-image-worker.service

# Main
systemctl enable --now local-ai-worker-tunnel.service
systemctl restart local-ai-web.service local-ai-telegram.service
systemctl disable --now local-ai-image.service local-ai-pose.service
```

The installer uses a shared `/srv/local-ai-worker/torch-runtime` copied from
the validated Main Python 3.12 runtime. This avoids duplicate CUDA libraries
and preserves the known-good PyTorch `2.13.0+cu129` binary set.

## Health And Logs

```bash
# Main service state
systemctl status local-ai-worker-tunnel local-ai-web local-ai-telegram

# AHN7 service and logs
ssh ahn7 systemctl status local-ai-image-worker
ssh ahn7 journalctl -u local-ai-image-worker -n 200 --no-pager

# Authenticated health (run on Main without printing the token)
set -a
source /etc/local-ai-agent/image-worker.env
set +a
curl -fsS -H "Authorization: Bearer ${IMAGE_WORKER_TOKEN}" \
  "${IMAGE_WORKER_URL}/health"
```

Health reports the active backend and PID, busy state, request/failure counts,
last error, GPU utilization, and VRAM. Journald records backend transitions and
capability latency.

## Failure And Recovery

If AHN7 or the tunnel is unavailable, image requests fail with HTTP 503 at the
Web layer. Main does not start SD-Turbo or LivePortrait. The tunnel and worker
units restart automatically. New requests recover after connectivity returns.

Rollback is explicit:

```bash
systemctl stop local-ai-worker-tunnel.service
systemctl enable --now local-ai-image.service local-ai-pose.service
```

Then remove `IMAGE_WORKER_URL` and `IMAGE_WORKER_TOKEN` from the Web/Telegram
service environment before restarting those services. Never enable local
services as an automatic failure fallback.