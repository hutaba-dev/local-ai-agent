from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from mcp import Client
from PIL import Image

from mcp_servers.media_server import create_media_mcp
from runtime.image_client import GeneratedImage, ImagePromptPlan, ImageQualityAssessment
from runtime.agent_runtime import AgentRuntime
from runtime.media import (
    AHN7MediaWorker,
    MediaAsset,
    MediaDirector,
    MediaError,
    MediaExecution,
    MediaOperation,
    MediaPlan,
    MediaResult,
    MediaSource,
    MediaStatus,
    MediaWorkerRegistry,
    VisualRequest,
    map_worker_error,
    validate_image,
)
from runtime.mcp_host import MCPHealth, MCPHost
from runtime.project_tools import ProjectTools
from runtime.projects import ProjectStore


def png_bytes(width: int = 512, height: int = 512) -> bytes:
    stream = BytesIO()
    Image.new("RGB", (width, height), "red").save(stream, "PNG")
    return stream.getvalue()


def execution(content: bytes | None = None) -> MediaExecution:
    image = content or png_bytes()
    request = VisualRequest("generate", "red square")
    plan = MediaPlan(request, (MediaOperation("image_generation", "red square", "image.generate"),), ())
    result = MediaResult(
        "mjob_123", "SUCCEEDED", "img_123", 512, 512, 42, "test-model", "worker-1",
        "generate", (), "2026-01-01T00:00:00+00:00",
    )
    return MediaExecution(result, image, plan, executed_capabilities=("image.generate",))


class MediaMCPTests(unittest.TestCase):
    @staticmethod
    def call(server: object, tool: str, arguments: dict[str, object]):
        async def invoke():
            async with Client(server, raise_exceptions=False) as client:
                return await client.call_tool(tool, arguments)

        return asyncio.run(invoke())

    def test_media_discovers_only_semantic_metadata_tools(self) -> None:
        scope = SimpleNamespace(owner_id="owner", project_id=None, conversation_id=None, tools=None)

        async def discover():
            async with Client(create_media_mcp(scope)) as client:
                return {tool.name: tool.input_schema for tool in (await client.list_tools()).tools}

        schemas = asyncio.run(discover())
        self.assertEqual(set(schemas), {
            "media_inspect_capability", "media_get_status", "media_generate_image",
            "media_edit_image", "media_adjust_pose",
        })
        serialized = json.dumps(schemas)
        for forbidden in ("owner_id", "project_id", "path", "base64", "AHN7", "RTX"):
            self.assertNotIn(forbidden, serialized)

    def test_generate_returns_logical_metadata_without_raw_bytes(self) -> None:
        scope = SimpleNamespace(owner_id="owner", project_id=None, conversation_id=None, tools=None)
        with patch("mcp_servers.media_server.MEDIA_DIRECTOR.execute", return_value=execution()):
            result = self.call(create_media_mcp(scope), "media_generate_image", {"subject": "red square"})

        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content["result"]["image_id"], "img_123")
        serialized = json.dumps(result.structured_content)
        self.assertNotIn("content", serialized)
        self.assertNotIn("base64", serialized)

    def test_non_project_generated_asset_is_scoped_to_the_injected_session(self) -> None:
        director = MediaDirector()
        director._assets["img_private"] = MediaAsset(
            "session-a", MediaSource("img_private", png_bytes(), "image/png"), "2026-01-01T00:00:00+00:00"
        )

        self.assertIsNotNone(director.asset("session-a", "img_private"))
        self.assertIsNone(director.asset("session-b", "img_private"))

    def test_multi_intent_plan_keeps_pose_before_generative_edit(self) -> None:
        director = MediaDirector()
        with patch("runtime.image_client.httpx.post", side_effect=OSError("planner offline")):
            plan, _, edit_plan = director.plan(VisualRequest(
                "edit", "portrait", "이 남자의 얼굴을 좀 더 잘생기게 하고 얼굴을 정면으로 바라보게 해줘.",
                source_image_ids=("fil_123",),
            ))

        self.assertIsNotNone(edit_plan)
        self.assertEqual(
            [operation.backend_capability for operation in plan.operations],
            ["portrait.frontalize", "image.edit"],
        )

    def test_unconfigured_registry_never_calls_local_image_endpoint(self) -> None:
        worker = AHN7MediaWorker()
        with patch("runtime.media.image_client.IMAGE_WORKER_URL", ""), patch(
            "runtime.media.image_client.IMAGE_WORKER_TOKEN", ""
        ), patch(
            "runtime.media.build_image_prompt",
            return_value=ImagePromptPlan("red square", "square", "still", "plain", "normal", "normal", ""),
        ), patch("runtime.media.image_client.httpx.post") as post:
            director = MediaDirector(MediaWorkerRegistry((worker,)))
            with self.assertRaises(MediaError) as raised:
                director.execute(VisualRequest("generate", "red square"))

        self.assertEqual(raised.exception.status, MediaStatus.UNCONFIGURED)
        post.assert_not_called()

    def test_malformed_worker_output_is_rejected(self) -> None:
        with self.assertRaises(MediaError) as raised:
            validate_image(b"not-an-image")
        self.assertEqual(raised.exception.status, MediaStatus.CAPABILITY_LIMITED)

    def test_worker_oom_response_is_not_collapsed_to_unavailable(self) -> None:
        request = Mock()
        response = Mock(status_code=502, text="backend task failed: CUDA out of memory")
        error = httpx.HTTPStatusError("failed", request=request, response=response)

        self.assertEqual(map_worker_error(error), MediaStatus.OOM)

    def test_project_source_is_owner_scoped_and_corrupt_images_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            (data_root / "projects").mkdir(parents=True)
            store = ProjectStore(root / "projects.db", data_root, require_mount=False)
            allowed_project = store.create_project("owner-a", "Allowed")
            denied_project = store.create_project("owner-b", "Denied")
            corrupt = store.save_file("owner-a", allowed_project["id"], "bad.png", b"bad", "image/png")
            denied = store.save_file("owner-b", denied_project["id"], "denied.png", png_bytes(), "image/png")
            scope = SimpleNamespace(
                owner_id="owner-a", project_id=allowed_project["id"], conversation_id=None,
                tools=ProjectTools(store),
            )
            server = create_media_mcp(scope)
            corrupt_result = self.call(server, "media_edit_image", {
                "source_image_id": corrupt["id"], "intent": "change background",
            })
            denied_result = self.call(server, "media_edit_image", {
                "source_image_id": denied["id"], "intent": "change background",
            })

        self.assertTrue(corrupt_result.is_error)
        self.assertTrue(denied_result.is_error)

    def test_project_artifact_records_media_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            (data_root / "projects").mkdir(parents=True)
            store = ProjectStore(root / "projects.db", data_root, require_mount=False)
            project = store.create_project("owner", "Media")
            scope = SimpleNamespace(
                owner_id="owner", project_id=project["id"], conversation_id=None, tools=ProjectTools(store),
            )
            with patch("mcp_servers.media_server.MEDIA_DIRECTOR.execute", return_value=execution()):
                result = self.call(create_media_mcp(scope), "media_generate_image", {
                    "subject": "red square", "save_to_project": True,
                })
            artifacts = store.list_artifacts("owner", project["id"])

        self.assertFalse(result.is_error)
        self.assertTrue(result.structured_content["result"]["artifact_id"].startswith("art_"))
        provenance = json.loads(artifacts[0]["description"])
        self.assertEqual(provenance["seed"], 42)
        self.assertEqual(provenance["worker"], "worker-1")
        self.assertEqual(provenance["executed_capabilities"], ["image.generate"])

    def test_host_preserves_busy_status_without_retry_or_fallback(self) -> None:
        host = MCPHost()
        with patch.dict(os.environ, {
            "MCP_ENABLED": "true", "MCP_MEDIA_ENABLED": "true",
        }, clear=False), patch("runtime.image_client.IMAGE_WORKER_URL", "http://127.0.0.1:18010"), patch(
            "runtime.image_client.IMAGE_WORKER_TOKEN", "configured"
        ), patch(
            "mcp_servers.media_server.MEDIA_DIRECTOR.execute",
            side_effect=MediaError(MediaStatus.BUSY, "media worker is busy"),
        ) as execute:
            outcome = host.call("media_generate_image", {"subject": "red square"})

        self.assertFalse(outcome.success)
        self.assertTrue(outcome.executed)
        self.assertEqual(outcome.status, MCPHealth.BUSY.value)
        execute.assert_called_once()

    def test_research_decision_supports_evidence_grounded_media_artifact(self) -> None:
        decision = AgentRuntime._parse_research_decision(json.dumps({
            "next_action": "CREATE_MEDIA",
            "queries": ["Evidence-grounded cutaway of a liquid hydrogen storage vessel"],
            "unresolved_questions": [],
            "decision_summary": "Evidence is sufficient for the requested visual.",
            "ready_to_answer": False,
            "save_to_project": True,
        }))

        self.assertEqual(decision.next_action, "CREATE_MEDIA")
        self.assertEqual(len(decision.queries), 1)
        self.assertTrue(decision.save_to_project)


if __name__ == "__main__":
    unittest.main()
