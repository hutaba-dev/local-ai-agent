"""Semantic media orchestration over the existing authenticated AHN7 worker."""

from __future__ import annotations

import io
import secrets
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from threading import Lock
from time import monotonic
from typing import Literal

import httpx
from PIL import Image, UnidentifiedImageError

from runtime import image_client
from runtime.image_client import (
    ImageEditCompletion,
    ImageEditPlan,
    ImagePromptPlan,
    ImageQualityAssessment,
    assess_image_edit_completion,
    assess_image_quality,
    build_image_edit_plan,
    build_image_prompt,
    correct_portrait_pose,
    create_image,
    edit_plan_prompt,
)


MAX_MEDIA_OPERATIONS = 3
MAX_MEDIA_IMAGE_BYTES = 20 * 1024 * 1024
MIN_MEDIA_DIMENSION = 16
MAX_MEDIA_DIMENSION = 4_096
HEALTH_CACHE_SECONDS = 10.0
MAX_TRACKED_JOBS = 100


class MediaStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"
    UNCONFIGURED = "UNCONFIGURED"
    BUSY = "BUSY"
    OOM = "OOM"
    TIMEOUT = "TIMEOUT"
    ERROR = "ERROR"
    MODEL_LIMITED = "MODEL_LIMITED"
    CAPABILITY_LIMITED = "CAPABILITY_LIMITED"
    PROJECT_STORAGE_OFFLINE = "PROJECT_STORAGE_OFFLINE"


class MediaJobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class MediaError(RuntimeError):
    def __init__(self, status: MediaStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class VisualRequest:
    operation: Literal["generate", "edit", "pose"]
    subject: str
    intent: str = ""
    composition: str = "unspecified"
    style: str = "unspecified"
    pose: str = "unspecified"
    expression: str = "unspecified"
    camera: str = "unspecified"
    lighting: str = "unspecified"
    background: str = "unspecified"
    quality_priority: Literal["FAST", "BALANCED", "QUALITY"] = "BALANCED"
    latency_priority: Literal["FAST", "BALANCED"] = "BALANCED"
    preserve_identity: bool = True
    preserve_elements: tuple[str, ...] = ()
    source_image_ids: tuple[str, ...] = ()
    output_size: str = "512x512"

    def instruction(self) -> str:
        values = {
            "subject": self.subject,
            "requested changes": self.intent,
            "composition": self.composition,
            "style": self.style,
            "pose": self.pose,
            "expression": self.expression,
            "camera": self.camera,
            "lighting": self.lighting,
            "background": self.background,
            "preserve": ", ".join(self.preserve_elements),
        }
        return ". ".join(f"{key}: {value}" for key, value in values.items() if value and value != "unspecified")


@dataclass(frozen=True)
class MediaOperation:
    type: str
    intent: str
    backend_capability: str


@dataclass(frozen=True)
class MediaPlan:
    request: VisualRequest
    operations: tuple[MediaOperation, ...]
    preservation_constraints: tuple[str, ...]
    output_requirements: tuple[str, ...] = ("valid decodable image", "512x512 PNG")


@dataclass(frozen=True)
class WorkerMetadata:
    worker_id: str
    health: str
    capabilities: tuple[str, ...]
    models: tuple[str, ...]
    max_resolution: str
    supports_edit: bool
    supports_pose: bool
    supports_reference: bool
    cost_class: str = "dedicated_gpu"
    latency_class: str = "long_running"


@dataclass(frozen=True)
class MediaResult:
    job_id: str
    status: str
    image_id: str
    width: int
    height: int
    seed: int
    model: str
    worker: str
    operation: str
    source_image_ids: tuple[str, ...]
    created_at: str
    warnings: tuple[str, ...] = ()
    artifact_id: str | None = None

    def normalized(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MediaSource:
    image_id: str
    content: bytes
    mime_type: str
    project_id: str | None = None


@dataclass(frozen=True)
class MediaAsset:
    namespace: str
    source: MediaSource
    created_at: str


@dataclass(frozen=True)
class MediaExecution:
    result: MediaResult
    content: bytes
    plan: MediaPlan
    prompt_plan: ImagePromptPlan | None = None
    edit_plan: ImageEditPlan | None = None
    edit_completion: ImageEditCompletion | None = None
    quality: ImageQualityAssessment | None = None
    executed_capabilities: tuple[str, ...] = ()


@dataclass(frozen=True)
class MediaEditExecution:
    generated: image_client.GeneratedImage
    prompt_plan: ImagePromptPlan
    edit_plan: ImageEditPlan
    completion: ImageEditCompletion
    executed_capabilities: tuple[str, ...]
    retry_count: int


@dataclass(frozen=True)
class MediaGenerationExecution:
    generated: image_client.GeneratedImage
    prompt_plan: ImagePromptPlan
    quality: ImageQualityAssessment
    candidate_reviews: tuple[dict[str, object], ...]
    retry_count: int


@dataclass
class MediaJob:
    job_id: str
    status: str
    operation: str
    created_at: str
    result: MediaResult | None = None
    error_status: str | None = None


class AHN7MediaWorker:
    worker_id = "ahn7"

    def __init__(self) -> None:
        self._health_lock = Lock()
        self._health_cached_at = 0.0
        self._health: WorkerMetadata | None = None

    def metadata(self, force: bool = False) -> WorkerMetadata:
        with self._health_lock:
            if not force and self._health is not None and monotonic() - self._health_cached_at < HEALTH_CACHE_SECONDS:
                return self._health
            self._health = self._probe()
            self._health_cached_at = monotonic()
            return self._health

    def _probe(self) -> WorkerMetadata:
        if not image_client.IMAGE_WORKER_URL or not image_client.IMAGE_WORKER_TOKEN:
            return WorkerMetadata(self.worker_id, MediaStatus.UNCONFIGURED.value, (), (), "512x512", True, True, True)
        headers = {"Authorization": f"Bearer {image_client.IMAGE_WORKER_TOKEN}"}
        try:
            health = httpx.get(f"{image_client.IMAGE_WORKER_URL}/health", headers=headers, timeout=5)
            capabilities = httpx.get(
                f"{image_client.IMAGE_WORKER_URL}/v1/capabilities", headers=headers, timeout=5
            )
            health.raise_for_status()
            capabilities.raise_for_status()
            health_payload = health.json()
            capability_payload = capabilities.json()
            values = tuple(str(item) for item in capability_payload.get("capabilities", []) if isinstance(item, str))
            models = tuple(str(item) for item in capability_payload.get("models", []) if isinstance(item, str))
            status = MediaStatus.BUSY.value if health_payload.get("busy") else MediaStatus.AVAILABLE.value
            return WorkerMetadata(
                self.worker_id,
                status,
                values,
                models or ("stabilityai/sd-turbo", "LivePortrait"),
                str(capability_payload.get("max_resolution", "512x512")),
                "image.edit" in values,
                "portrait.frontalize" in values,
                "image.edit" in values,
            )
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            return WorkerMetadata(
                self.worker_id, MediaStatus.UNAVAILABLE.value, (), (), "512x512", False, False, False
            )

    @staticmethod
    def generate(prompt: str) -> image_client.GeneratedImage:
        return create_image(prompt)

    @staticmethod
    def edit(prompt: str, source: bytes, strength: float = 0.5) -> image_client.GeneratedImage:
        return create_image(prompt, source, strength)

    @staticmethod
    def adjust_pose(source: bytes) -> image_client.GeneratedImage:
        return correct_portrait_pose(source)


class MediaWorkerRegistry:
    def __init__(self, workers: tuple[AHN7MediaWorker, ...] | None = None) -> None:
        self._workers = workers or (AHN7MediaWorker(),)

    def select(self, capability: str) -> AHN7MediaWorker:
        for worker in self._workers:
            metadata = worker.metadata()
            if metadata.health == MediaStatus.AVAILABLE.value and capability in metadata.capabilities:
                return worker
            if metadata.health == MediaStatus.BUSY.value and capability in metadata.capabilities:
                raise MediaError(MediaStatus.BUSY, "media worker is busy")
        statuses = {worker.metadata().health for worker in self._workers}
        if MediaStatus.UNCONFIGURED.value in statuses:
            raise MediaError(MediaStatus.UNCONFIGURED, "media worker is not configured")
        raise MediaError(MediaStatus.UNAVAILABLE, "media capability is unavailable")

    def compact_status(self) -> dict[str, object]:
        workers = [worker.metadata() for worker in self._workers]
        available = next((worker for worker in workers if worker.health == MediaStatus.AVAILABLE.value), None)
        selected = available or workers[0]
        return asdict(selected)


class MediaDirector:
    def __init__(self, registry: MediaWorkerRegistry | None = None) -> None:
        self.registry = registry or MediaWorkerRegistry()
        self._jobs: dict[str, MediaJob] = {}
        self._job_lock = Lock()
        self._assets: dict[str, MediaAsset] = {}

    def status(self, force: bool = False) -> dict[str, object]:
        if force:
            for worker in self.registry._workers:
                worker.metadata(force=True)
        return self.registry.compact_status()

    def job(self, job_id: str) -> dict[str, object] | None:
        with self._job_lock:
            job = self._jobs.get(job_id)
            return asdict(job) if job else None

    def asset(self, namespace: str, image_id: str) -> MediaSource | None:
        with self._job_lock:
            asset = self._assets.get(image_id)
            return asset.source if asset and secrets.compare_digest(asset.namespace, namespace) else None

    def plan(self, request: VisualRequest) -> tuple[MediaPlan, ImagePromptPlan | None, ImageEditPlan | None]:
        if request.output_size != "512x512":
            raise MediaError(MediaStatus.MODEL_LIMITED, "current worker supports only 512x512 output")
        if request.operation == "generate":
            prompt_plan = build_image_prompt(request.instruction())
            return MediaPlan(
                request,
                (MediaOperation("image_generation", request.instruction(), "image.generate"),),
                request.preserve_elements,
            ), prompt_plan, None
        if not request.source_image_ids:
            raise MediaError(MediaStatus.CAPABILITY_LIMITED, "image edit and pose adjustment require a source image")
        if request.operation == "pose":
            if request.pose not in {"front_facing", "unspecified"}:
                raise MediaError(MediaStatus.MODEL_LIMITED, "current pose backend supports front-facing correction only")
            return MediaPlan(
                request,
                (MediaOperation("pose_adjustment", "front-facing portrait correction", "portrait.frontalize"),),
                request.preserve_elements,
            ), None, None
        edit_plan = build_image_edit_plan(request.intent or request.instruction())
        edits = edit_plan.edits[:MAX_MEDIA_OPERATIONS]
        if len(edit_plan.edits) > MAX_MEDIA_OPERATIONS:
            raise MediaError(MediaStatus.CAPABILITY_LIMITED, "media plan exceeds the operation limit")
        operations = tuple(MediaOperation(edit.type, edit.instruction, (
            "portrait.frontalize" if edit.capability == "pose_correction" else "image.edit"
        )) for edit in edits)
        prompt_plan = build_image_prompt(
            edit_plan_prompt(edit_plan), editing=True, original_request=request.intent or request.instruction()
        )
        return MediaPlan(request, operations, edit_plan.constraints), prompt_plan, edit_plan

    def execute(
        self, request: VisualRequest, source: MediaSource | None = None, *, namespace: str = "runtime"
    ) -> MediaExecution:
        job_id = f"mjob_{secrets.token_hex(16)}"
        created_at = datetime.now(UTC).isoformat()
        self._set_job(MediaJob(job_id, MediaJobStatus.QUEUED.value, request.operation, created_at))
        try:
            self._set_job(MediaJob(job_id, MediaJobStatus.RUNNING.value, request.operation, created_at))
            plan, prompt_plan, edit_plan = self.plan(request)
            source_metadata = validate_image(source.content) if source is not None else None
            if request.operation != "generate" and source is None:
                raise MediaError(MediaStatus.CAPABILITY_LIMITED, "source image is required")
            warnings = self._priority_warnings(request)
            executed: list[str] = []
            selected_worker: AHN7MediaWorker | None = None
            completion = None
            quality = None
            if request.operation == "generate":
                worker = self.registry.select("image.generate")
                selected_worker = worker
                generation = execute_media_generation(
                    request.instruction(),
                    prompt_plan=prompt_plan,
                    prompt_builder=build_image_prompt,
                    image_executor=lambda generated_prompt, _source=None: self._worker_call(
                        worker.generate, generated_prompt
                    ),
                    quality_assessor=assess_image_quality,
                    candidate_count=2 if prompt_plan and prompt_plan.quality_sensitive else 1,
                )
                generated = generation.generated
                prompt_plan = generation.prompt_plan
                quality = generation.quality
                executed.append("image.generate")
                if generation.retry_count:
                    executed.append("image.generate.retry")
            elif request.operation == "pose":
                worker = self.registry.select("portrait.frontalize")
                selected_worker = worker
                generated = self._worker_call(worker.adjust_pose, source.content)
                executed.append("portrait.frontalize")
            else:
                assert source is not None and edit_plan is not None
                pose_worker = self.registry.select("portrait.frontalize") if any(
                    operation.backend_capability == "portrait.frontalize" for operation in plan.operations
                ) else None
                edit_worker = self.registry.select("image.edit") if any(
                    operation.backend_capability == "image.edit" for operation in plan.operations
                ) else None
                selected_worker = edit_worker or pose_worker
                edit_execution = execute_media_edit(
                    request.intent or request.instruction(),
                    source.content,
                    edit_plan=edit_plan,
                    prompt_plan=prompt_plan,
                    prompt_builder=build_image_prompt,
                    pose_executor=(lambda content: self._worker_call(pose_worker.adjust_pose, content)) if pose_worker else None,
                    edit_executor=(lambda edit_prompt, content, strength=0.5: self._worker_call(
                        edit_worker.edit, edit_prompt, content, strength
                    )) if edit_worker else None,
                    completion_assessor=assess_image_edit_completion,
                )
                generated = edit_execution.generated
                completion = edit_execution.completion
                executed.extend(edit_execution.executed_capabilities)
            image_metadata = validate_image(generated.content)
            assert selected_worker is not None
            model = "LivePortrait" if generated.mode == "pose" else "stabilityai/sd-turbo"
            if source_metadata and (source_metadata[0], source_metadata[1]) != (image_metadata[0], image_metadata[1]):
                warnings += ("output dimensions differ from normalized source dimensions",)
            result = MediaResult(
                job_id, MediaJobStatus.SUCCEEDED.value, f"img_{secrets.token_hex(16)}",
                image_metadata[0], image_metadata[1], generated.seed, model, selected_worker.worker_id,
                "multi_step_edit" if len(plan.operations) > 1 else request.operation,
                request.source_image_ids, created_at, warnings,
            )
            execution = MediaExecution(
                result, generated.content, plan, prompt_plan, edit_plan, completion, quality, tuple(executed)
            )
            self._set_job(
                MediaJob(job_id, MediaJobStatus.SUCCEEDED.value, request.operation, created_at, result),
                MediaAsset(namespace, MediaSource(result.image_id, generated.content, image_metadata[2]), created_at),
            )
            return execution
        except MediaError as error:
            self._set_job(MediaJob(job_id, MediaJobStatus.FAILED.value, request.operation, created_at, error_status=error.status.value))
            raise
        except Exception as error:
            status = map_worker_error(error)
            self._set_job(MediaJob(job_id, MediaJobStatus.FAILED.value, request.operation, created_at, error_status=status.value))
            raise MediaError(status, "media operation failed") from error

    @staticmethod
    def _worker_call(function, *args):
        try:
            return function(*args)
        except Exception as error:
            raise MediaError(map_worker_error(error), "media worker operation failed") from error

    @staticmethod
    def _priority_warnings(request: VisualRequest) -> tuple[str, ...]:
        warnings = []
        if request.quality_priority != "BALANCED":
            warnings.append("MODEL_LIMITED: current backend uses one fixed quality route")
        if request.latency_priority != "BALANCED":
            warnings.append("CAPABILITY_LIMITED: current backend does not provide a separate latency route")
        return tuple(warnings)

    def _set_job(self, job: MediaJob, asset: MediaAsset | None = None) -> None:
        with self._job_lock:
            self._jobs[job.job_id] = job
            if asset is not None:
                self._assets[asset.source.image_id] = asset
            while len(self._jobs) > MAX_TRACKED_JOBS:
                self._jobs.pop(next(iter(self._jobs)))
            while len(self._assets) > MAX_TRACKED_JOBS:
                self._assets.pop(next(iter(self._assets)))


def validate_image(content: bytes) -> tuple[int, int, str]:
    if not content or len(content) > MAX_MEDIA_IMAGE_BYTES:
        raise MediaError(MediaStatus.CAPABILITY_LIMITED, "image size is invalid")
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
        with Image.open(io.BytesIO(content)) as image:
            width, height = image.size
            image_format = str(image.format or "").upper()
    except (OSError, UnidentifiedImageError) as error:
        raise MediaError(MediaStatus.CAPABILITY_LIMITED, "image is not decodable") from error
    if image_format not in {"PNG", "JPEG", "WEBP"}:
        raise MediaError(MediaStatus.CAPABILITY_LIMITED, "image format is unsupported")
    if not MIN_MEDIA_DIMENSION <= width <= MAX_MEDIA_DIMENSION or not MIN_MEDIA_DIMENSION <= height <= MAX_MEDIA_DIMENSION:
        raise MediaError(MediaStatus.CAPABILITY_LIMITED, "image dimensions are unsupported")
    return width, height, f"image/{'jpeg' if image_format == 'JPEG' else image_format.lower()}"


def execute_media_edit(
    intent: str,
    source: bytes,
    *,
    edit_plan: ImageEditPlan | None = None,
    prompt_plan: ImagePromptPlan | None = None,
    plan_builder=build_image_edit_plan,
    prompt_builder=build_image_prompt,
    pose_executor=correct_portrait_pose,
    edit_executor=create_image,
    completion_assessor=assess_image_edit_completion,
) -> MediaEditExecution:
    resolved_plan = edit_plan or plan_builder(intent)
    planned_prompt = edit_plan_prompt(resolved_plan)
    resolved_prompt = prompt_plan or prompt_builder(planned_prompt, editing=True, original_request=intent)
    generated = image_client.GeneratedImage(source, 0, "edit", "edit-intermediate.png")
    executed: list[str] = []
    if any(edit.capability == "pose_correction" for edit in resolved_plan.edits):
        if pose_executor is None:
            raise MediaError(MediaStatus.CAPABILITY_LIMITED, "pose adjustment is unavailable")
        generated = pose_executor(generated.content)
        executed.append("portrait.frontalize")
    generative_source = generated.content
    has_generative_edit = any(edit.capability == "generative_edit" for edit in resolved_plan.edits)
    appearance_sensitive = any(edit.type == "appearance_refinement" for edit in resolved_plan.edits)
    if has_generative_edit:
        if edit_executor is None:
            raise MediaError(MediaStatus.CAPABILITY_LIMITED, "generative edit is unavailable")
        generated = (
            edit_executor(resolved_prompt.prompt, generated.content, 0.25)
            if appearance_sensitive else edit_executor(resolved_prompt.prompt, generated.content)
        )
        executed.append("image.edit")
    completion = completion_assessor(resolved_plan, source, generated.content)
    retry_count = 0
    if completion.checked and not completion.passed:
        pending = [edit_type for edit_type, complete in completion.edit_status if not complete]
        if resolved_plan.preserve_identity and not completion.identity_preserved:
            pending.append("identity_preservation")
        retry_prompt = f"{planned_prompt} Retry incomplete edits: {', '.join(pending)}."
        retry_plan = prompt_builder(retry_prompt, editing=True, original_request=intent)
        if has_generative_edit:
            retry_source = generative_source if "identity_preservation" in pending else generated.content
            generated = (
                edit_executor(retry_plan.prompt, retry_source, 0.25)
                if appearance_sensitive else edit_executor(retry_plan.prompt, retry_source)
            )
            executed.append("image.edit.retry")
        else:
            generated = pose_executor(source)
            executed.append("portrait.frontalize.retry")
        completion = completion_assessor(resolved_plan, source, generated.content)
        retry_count = 1
    return MediaEditExecution(
        generated, resolved_prompt, resolved_plan, completion, tuple(executed), retry_count
    )


def execute_media_generation(
    intent: str,
    *,
    source: bytes | None = None,
    prompt_plan: ImagePromptPlan | None = None,
    prompt_builder=build_image_prompt,
    image_executor=create_image,
    quality_assessor=assess_image_quality,
    candidate_count: int = 1,
    preference_context: str = "",
    reference_cues: str = "",
) -> MediaGenerationExecution:
    resolved_prompt = prompt_plan or prompt_builder(
        intent, preference_context=preference_context, reference_cues=reference_cues
    )
    bounded_candidates = max(1, min(candidate_count, 2))
    candidates: list[tuple[image_client.GeneratedImage, ImageQualityAssessment]] = []
    reviews: list[dict[str, object]] = []
    for candidate_number in range(1, bounded_candidates + 1):
        candidate = image_executor(resolved_prompt.prompt, source)
        quality = quality_assessor(intent, candidate.content)
        candidates.append((candidate, quality))
        reviews.append({
            "candidate": candidate_number,
            "seed": candidate.seed,
            "score": quality.overall_score,
            "passed": quality.passed,
            "failures": list(quality.failures),
        })
    generated, quality = max(
        candidates,
        key=lambda candidate: (
            candidate[1].passed,
            candidate[1].overall_score if candidate[1].checked else 0,
        ),
    )
    retry_count = 1 if bounded_candidates > 1 else 0
    if bounded_candidates == 1 and quality.checked and not quality.passed:
        failure_summary = ", ".join(quality.failures)
        if quality.summary:
            failure_summary = f"{failure_summary}. {quality.summary}"
        resolved_prompt = prompt_builder(
            intent,
            original_request=intent,
            quality_feedback=failure_summary,
            simplify_composition=True,
            preference_context=preference_context,
            reference_cues=reference_cues,
        )
        generated = image_executor(resolved_prompt.prompt, None)
        quality = quality_assessor(intent, generated.content)
        reviews.append({
            "candidate": 2,
            "seed": generated.seed,
            "score": quality.overall_score,
            "passed": quality.passed,
            "failures": list(quality.failures),
        })
        retry_count = 1
    return MediaGenerationExecution(generated, resolved_prompt, quality, tuple(reviews), retry_count)


def map_worker_error(error: Exception) -> MediaStatus:
    if isinstance(error, MediaError):
        return error.status
    if isinstance(error, httpx.TimeoutException):
        return MediaStatus.TIMEOUT
    if isinstance(error, httpx.HTTPStatusError):
        status_code = error.response.status_code
        text = error.response.text.lower()
        if "out of memory" in text or "oom" in text:
            return MediaStatus.OOM
        if status_code == 429:
            return MediaStatus.BUSY
        if status_code in {502, 503}:
            return MediaStatus.UNAVAILABLE
        if status_code == 504:
            return MediaStatus.TIMEOUT
    if isinstance(error, (httpx.NetworkError, OSError)):
        return MediaStatus.UNAVAILABLE
    return MediaStatus.ERROR


MEDIA_DIRECTOR = MediaDirector()
