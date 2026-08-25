"""Authenticated capability-based worker with one resident GPU backend."""

from __future__ import annotations

import atexit
import base64
import io
import logging
import os
import secrets
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from PIL import Image
from pydantic import BaseModel, Field, ValidationError


WORKER_TOKEN = os.getenv("IMAGE_WORKER_TOKEN", "")
PROJECT_ROOT = Path(os.getenv("IMAGE_WORKER_PROJECT_ROOT", "/srv/local-ai-worker/app"))
MAX_REQUEST_BYTES = int(os.getenv("IMAGE_WORKER_MAX_REQUEST_BYTES", str(24 * 1024 * 1024)))
MAX_IMAGE_BYTES = int(os.getenv("IMAGE_WORKER_MAX_IMAGE_BYTES", str(15 * 1024 * 1024)))
QUEUE_TIMEOUT_SECONDS = float(os.getenv("IMAGE_WORKER_QUEUE_TIMEOUT_SECONDS", "10"))
BACKEND_START_TIMEOUT_SECONDS = float(os.getenv("IMAGE_WORKER_BACKEND_START_TIMEOUT_SECONDS", "180"))
TASK_TIMEOUT_SECONDS = float(os.getenv("IMAGE_WORKER_TASK_TIMEOUT_SECONDS", "300"))
MAX_BASE64_LENGTH = ((MAX_IMAGE_BYTES + 2) // 3) * 4
LOGGER = logging.getLogger("uvicorn.error")


class ImageTask(BaseModel):
    prompt: str = Field(min_length=1, max_length=2_000)
    source_image_base64: str | None = Field(default=None, max_length=MAX_BASE64_LENGTH)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    strength: float = Field(default=0.5, ge=0.5, le=0.95)


class PoseTask(BaseModel):
    source_image_base64: str = Field(min_length=1, max_length=MAX_BASE64_LENGTH)


@dataclass(frozen=True)
class BackendSpec:
    name: str
    python: str
    port: int


@dataclass(frozen=True)
class BackendResult:
    content: bytes
    media_type: str
    headers: dict[str, str]


class WorkerBusyError(RuntimeError):
    pass


class BackendUnavailableError(RuntimeError):
    pass


BACKENDS = {
    "image": BackendSpec(
        "image",
        os.getenv("IMAGE_BACKEND_PYTHON", "/srv/local-ai-worker/image-venv/bin/python"),
        int(os.getenv("IMAGE_BACKEND_PORT", "18001")),
    ),
    "pose": BackendSpec(
        "pose",
        os.getenv("POSE_BACKEND_PYTHON", "/srv/local-ai-worker/pose-venv/bin/python"),
        int(os.getenv("POSE_BACKEND_PORT", "18002")),
    ),
}

CAPABILITIES = {
    "image.generate": ("image", "/v1/image"),
    "image.edit": ("image", "/v1/image"),
    "portrait.frontalize": ("pose", "/v1/frontalize"),
}


class BackendManager:
    def __init__(self) -> None:
        self._lock = Lock()
        self._state_lock = Lock()
        self._process: subprocess.Popen[bytes] | None = None
        self._active_backend: str | None = None
        self._requests = 0
        self._failures = 0
        self._last_error: str | None = None

    def execute(self, capability: str, payload: dict[str, object]) -> BackendResult:
        if not self._lock.acquire(timeout=QUEUE_TIMEOUT_SECONDS):
            LOGGER.warning("worker_busy capability=%s", capability)
            raise WorkerBusyError("GPU worker is busy")
        started = time.monotonic()
        try:
            backend_name, path = CAPABILITIES[capability]
            self._activate(backend_name)
            spec = BACKENDS[backend_name]
            response = httpx.post(
                f"http://127.0.0.1:{spec.port}{path}",
                json=payload,
                timeout=TASK_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            headers = {
                key: value
                for key, value in response.headers.items()
                if key.lower().startswith("x-image-") or key.lower().startswith("x-source-")
            }
            with self._state_lock:
                self._requests += 1
                self._last_error = None
            LOGGER.info(
                "task_complete capability=%s backend=%s duration_seconds=%.3f bytes=%d",
                capability,
                backend_name,
                time.monotonic() - started,
                len(response.content),
            )
            return BackendResult(response.content, response.headers.get("content-type", "image/png"), headers)
        except (httpx.HTTPError, OSError, RuntimeError) as error:
            with self._state_lock:
                self._failures += 1
                self._last_error = str(error)[:500]
            LOGGER.exception(
                "task_failed capability=%s duration_seconds=%.3f",
                capability,
                time.monotonic() - started,
            )
            raise
        finally:
            self._lock.release()

    def _activate(self, backend_name: str) -> None:
        if self._active_backend == backend_name and self._process is not None and self._process.poll() is None:
            return
        self.stop()
        spec = BACKENDS[backend_name]
        LOGGER.info("backend_start backend=%s port=%d", backend_name, spec.port)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PROJECT_ROOT)
        command = [
            spec.python,
            "-m",
            "uvicorn",
            f"{spec.name}_service.app:app" if spec.name == "image" else "pose_service.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(spec.port),
        ]
        try:
            self._process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment)
        except OSError as error:
            raise BackendUnavailableError(f"could not start {backend_name} backend: {error}") from error
        self._active_backend = backend_name
        deadline = time.monotonic() + BACKEND_START_TIMEOUT_SECONDS
        health_url = f"http://127.0.0.1:{spec.port}/health"
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                self.stop()
                raise BackendUnavailableError(f"{backend_name} backend exited during startup")
            try:
                response = httpx.get(health_url, timeout=2)
                if response.status_code == 200:
                    LOGGER.info("backend_ready backend=%s pid=%d", backend_name, self._process.pid)
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        self.stop()
        raise BackendUnavailableError(f"{backend_name} backend startup timed out")

    def stop(self) -> None:
        process = self._process
        backend_name = self._active_backend
        self._process = None
        self._active_backend = None
        if process is None or process.poll() is not None:
            return
        LOGGER.info("backend_stop backend=%s pid=%d", backend_name, process.pid)
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)

    def status(self) -> dict[str, object]:
        process = self._process
        with self._state_lock:
            return {
                "active_backend": self._active_backend,
                "backend_pid": process.pid if process is not None and process.poll() is None else None,
                "busy": self._lock.locked(),
                "requests": self._requests,
                "failures": self._failures,
                "last_error": self._last_error,
            }


def _authorized(authorization: str | None) -> None:
    if not WORKER_TOKEN:
        raise HTTPException(status_code=503, detail="worker token is not configured")
    expected = f"Bearer {WORKER_TOKEN}"
    if authorization is None or not secrets.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="invalid bearer token")


def _validated_image(encoded: str) -> None:
    try:
        content = base64.b64decode(encoded, validate=True)
        if len(content) > MAX_IMAGE_BYTES:
            raise ValueError("source image exceeds size limit")
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
            if image.format not in {"PNG", "JPEG", "WEBP"}:
                raise ValueError("source image must be PNG, JPEG, or WebP")
    except (ValueError, OSError) as error:
        raise HTTPException(status_code=422, detail=str(error) or "source image is invalid") from error


def _gpu_status() -> dict[str, object]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        name, total, used, utilization = [value.strip() for value in result.stdout.strip().split(",")]
        return {
            "name": name,
            "memory_total_mib": int(total),
            "memory_used_mib": int(used),
            "utilization_percent": int(utilization),
        }
    except (OSError, ValueError, subprocess.SubprocessError):
        return {"status": "unavailable"}


backend_manager = BackendManager()
atexit.register(backend_manager.stop)
app = FastAPI(title="Image Capability Worker", docs_url=None, redoc_url=None)


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > MAX_REQUEST_BYTES:
                return JSONResponse({"detail": "request exceeds size limit"}, status_code=413)
        except ValueError:
            return JSONResponse({"detail": "invalid content-length"}, status_code=400)
    body = await request.body()
    if len(body) > MAX_REQUEST_BYTES:
        return JSONResponse({"detail": "request exceeds size limit"}, status_code=413)
    return await call_next(request)


@app.get("/health")
def health(authorization: str | None = Header(default=None)) -> dict[str, object]:
    _authorized(authorization)
    return {"status": "ok", **backend_manager.status(), "gpu": _gpu_status()}


@app.get("/v1/capabilities")
def capabilities(authorization: str | None = Header(default=None)) -> dict[str, object]:
    _authorized(authorization)
    return {"capabilities": sorted(CAPABILITIES), "max_concurrency": 1}


@app.post("/v1/tasks/{capability}")
def task(
    capability: str,
    payload: dict[str, object],
    authorization: str | None = Header(default=None),
) -> Response:
    _authorized(authorization)
    if capability not in CAPABILITIES:
        raise HTTPException(status_code=404, detail="unsupported capability")
    try:
        if capability in {"image.generate", "image.edit"}:
            validated = ImageTask.model_validate(payload)
            if capability == "image.generate" and validated.source_image_base64 is not None:
                raise HTTPException(status_code=422, detail="image.generate does not accept a source image")
            if capability == "image.edit" and validated.source_image_base64 is None:
                raise HTTPException(status_code=422, detail="image.edit requires a source image")
            if validated.source_image_base64 is not None:
                _validated_image(validated.source_image_base64)
        else:
            validated = PoseTask.model_validate(payload)
            _validated_image(validated.source_image_base64)
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=error.errors(include_url=False)) from error
    try:
        result = backend_manager.execute(capability, validated.model_dump(exclude_none=True))
    except WorkerBusyError as error:
        raise HTTPException(status_code=429, detail=str(error)) from error
    except BackendUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except httpx.TimeoutException as error:
        raise HTTPException(status_code=504, detail="backend task timed out") from error
    except (httpx.HTTPError, OSError, RuntimeError) as error:
        raise HTTPException(status_code=502, detail=f"backend task failed: {error}") from error
    return Response(result.content, media_type=result.media_type, headers=result.headers)