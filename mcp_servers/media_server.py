"""Request-scoped semantic MCP facade for the existing Media runtime."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Literal, Protocol

from mcp.server import MCPServer

from runtime.media import MEDIA_DIRECTOR, MediaError, MediaResult, MediaSource, VisualRequest, validate_image
from runtime.projects import ProjectStorageOfflineError


class MediaScope(Protocol):
    owner_id: str
    project_id: str | None
    conversation_id: str | None
    tools: Any


_SOURCE_PREFIXES = ("img_", "fil_")


def create_media_mcp(scope: MediaScope) -> MCPServer:
    server = MCPServer(
        "ahnbys-media",
        description="Semantic image generation, editing, pose adjustment, status, and durable artifact storage.",
        version="1.0.0",
    )
    namespace = f"{scope.owner_id}:{scope.project_id or 'personal'}"

    def source_by_id(image_id: str) -> MediaSource:
        if not image_id.startswith(_SOURCE_PREFIXES) or len(image_id) > 100:
            raise ValueError("source_image_id must be an opaque image or project file ID")
        generated = MEDIA_DIRECTOR.asset(namespace, image_id)
        if generated is not None:
            return generated
        if not image_id.startswith("fil_") or not scope.project_id:
            raise ValueError("source image is unavailable or unauthorized")
        metadata, content = scope.tools.store.read_file(scope.owner_id, scope.project_id, image_id)
        mime_type = str(metadata.get("mime_type", ""))
        if not mime_type.startswith("image/"):
            raise ValueError("project file is not an image")
        width, height, normalized_mime = validate_image(content)
        if width < 16 or height < 16:
            raise ValueError("source image is too small")
        return MediaSource(image_id, content, normalized_mime, scope.project_id)

    def artifact_result(execution, save_to_project: bool) -> MediaResult:
        if not save_to_project:
            return execution.result
        if not scope.project_id:
            raise ValueError("save_to_project requires an authorized current project")
        try:
            scope.tools.store.require_storage()
        except ProjectStorageOfflineError:
            raise
        provenance = {
            "media_operation": execution.result.operation,
            "worker": execution.result.worker,
            "model": execution.result.model,
            "seed": execution.result.seed,
            "source_image_ids": list(execution.result.source_image_ids),
            "executed_capabilities": list(execution.executed_capabilities),
            "created_at": execution.result.created_at,
        }
        artifact = scope.tools.store.save_file(
            scope.owner_id,
            scope.project_id,
            f"media-{execution.result.job_id}.png",
            execution.content,
            "image/png",
            conversation_id=scope.conversation_id,
            artifact=True,
            creator="media",
            description=json.dumps(provenance, ensure_ascii=True, separators=(",", ":")),
        )
        return replace(execution.result, artifact_id=str(artifact.get("artifact_id") or "") or None)

    def response(execution, save_to_project: bool) -> dict[str, object]:
        result = artifact_result(execution, save_to_project)
        return {
            "status": "AVAILABLE",
            "result": result.normalized(),
            "plan": {
                "operations": [
                    {"type": operation.type, "intent": operation.intent}
                    for operation in execution.plan.operations
                ],
                "preservation_constraints": list(execution.plan.preservation_constraints),
                "output_requirements": list(execution.plan.output_requirements),
            },
        }

    def execute(request: VisualRequest, source: MediaSource | None, save_to_project: bool) -> dict[str, object]:
        try:
            execution = MEDIA_DIRECTOR.execute(request, source, namespace=namespace)
            return response(execution, save_to_project)
        except MediaError as error:
            return {"status": error.status.value, "error": str(error)}
        except ProjectStorageOfflineError:
            return {"status": "PROJECT_STORAGE_OFFLINE", "error": "project storage is offline"}

    @server.tool(
        description="Inspect current semantic image operations, model limits, and worker availability without executing a job.",
        structured_output=True,
    )
    def media_inspect_capability() -> dict[str, object]:
        status = MEDIA_DIRECTOR.status()
        return {
            "status": status["health"],
            "operations": ["generate", "edit", "pose"],
            "limits": {
                "max_operations": 3,
                "output_sizes": [status["max_resolution"]],
                "supports_edit": status["supports_edit"],
                "supports_pose": status["supports_pose"],
                "supports_reference": status["supports_reference"],
            },
        }

    @server.tool(description="Get current image capability health or the normalized state of one recent job.", structured_output=True)
    def media_get_status(job_id: str | None = None) -> dict[str, object]:
        if job_id is None:
            status = MEDIA_DIRECTOR.status()
            return {"status": status["health"], "worker": status}
        if not job_id.startswith("mjob_") or len(job_id) > 100:
            raise ValueError("invalid media job ID")
        job = MEDIA_DIRECTOR.job(job_id)
        return {"status": "UNAVAILABLE", "job_id": job_id} if job is None else {"status": "AVAILABLE", "job": job}

    @server.tool(
        description="Generate one image from explicit visual intent. Returns metadata and a logical image ID, never raw image bytes.",
        structured_output=True,
    )
    def media_generate_image(
        subject: str,
        intent: str = "",
        composition: str = "unspecified",
        style: str = "unspecified",
        camera: str = "unspecified",
        lighting: str = "unspecified",
        background: str = "unspecified",
        quality_priority: Literal["FAST", "BALANCED", "QUALITY"] = "BALANCED",
        latency_priority: Literal["FAST", "BALANCED"] = "BALANCED",
        output_size: str = "512x512",
        save_to_project: bool = False,
    ) -> dict[str, object]:
        if not 1 <= len(subject.strip()) <= 500 or len(intent) > 2_000:
            raise ValueError("subject or intent length is invalid")
        request = VisualRequest(
            "generate", subject.strip(), intent.strip(), composition[:300], style[:300],
            camera=camera[:300], lighting=lighting[:300], background=background[:300],
            quality_priority=quality_priority, latency_priority=latency_priority, output_size=output_size,
        )
        return execute(request, None, save_to_project)

    @server.tool(
        description="Apply all requested changes to one authorized logical source image while preserving identity and unchanged elements.",
        structured_output=True,
    )
    def media_edit_image(
        source_image_id: str,
        intent: str,
        preserve_identity: bool = True,
        preserve_elements: list[str] | None = None,
        quality_priority: Literal["FAST", "BALANCED", "QUALITY"] = "BALANCED",
        latency_priority: Literal["FAST", "BALANCED"] = "BALANCED",
        output_size: str = "512x512",
        save_to_project: bool = False,
    ) -> dict[str, object]:
        if not 1 <= len(intent.strip()) <= 2_000:
            raise ValueError("edit intent length is invalid")
        preserved = tuple(item[:200] for item in (preserve_elements or [])[:20] if item.strip())
        request = VisualRequest(
            "edit", "source image", intent.strip(), preserve_identity=preserve_identity,
            preserve_elements=preserved, source_image_ids=(source_image_id,),
            quality_priority=quality_priority, latency_priority=latency_priority, output_size=output_size,
        )
        return execute(request, source_by_id(source_image_id), save_to_project)

    @server.tool(
        description="Adjust an authorized portrait source image to a front-facing pose while preserving identity.",
        structured_output=True,
    )
    def media_adjust_pose(
        source_image_id: str,
        pose: Literal["front_facing"] = "front_facing",
        preserve_identity: bool = True,
        save_to_project: bool = False,
    ) -> dict[str, object]:
        request = VisualRequest(
            "pose", "portrait", pose=pose, preserve_identity=preserve_identity,
            source_image_ids=(source_image_id,),
        )
        return execute(request, source_by_id(source_image_id), save_to_project)

    return server
