import base64
import unittest
from io import BytesIO
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from PIL import Image

from image_service import app as image_app
from runtime import image_client
from runtime.image_client import parse_image_command, prefers_original_source


def png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (32, 24), "red").save(buffer, format="PNG")
    return buffer.getvalue()


class ImageServiceTests(unittest.TestCase):
    def test_remote_worker_uses_capability_and_bearer_token_without_local_fallback(self) -> None:
        response = Mock()
        response.content = png_bytes()
        response.headers = {"X-Image-Seed": "42", "X-Image-Mode": "generate"}
        response.raise_for_status.return_value = None

        with patch.object(image_client, "IMAGE_WORKER_URL", "http://worker:8010"), patch.object(
            image_client, "IMAGE_WORKER_TOKEN", "secret"
        ), patch("runtime.image_client.httpx.post", return_value=response) as post:
            generated = image_client.create_image("red square")

        self.assertEqual(generated.seed, 42)
        self.assertEqual(post.call_args.args[0], "http://worker:8010/v1/tasks/image.generate")
        self.assertEqual(post.call_args.kwargs["headers"], {"Authorization": "Bearer secret"})
        self.assertNotIn("127.0.0.1", post.call_args.args[0])

    def test_remote_worker_requires_token(self) -> None:
        with patch.object(image_client, "IMAGE_WORKER_URL", "http://worker:8010"), patch.object(
            image_client, "IMAGE_WORKER_TOKEN", ""
        ), patch("runtime.image_client.httpx.post") as post:
            with self.assertRaisesRegex(RuntimeError, "IMAGE_WORKER_TOKEN"):
                image_client.correct_portrait_pose(png_bytes())
        post.assert_not_called()

    def test_parses_generation_and_edit_commands(self) -> None:
        self.assertEqual(parse_image_command("/image a glass tower"), ("image", "a glass tower"))
        self.assertEqual(parse_image_command("/edit  배경을 밤으로 "), ("edit", "배경을 밤으로"))
        natural_edit = "고개를 똑바로 보게 바꿔주고, 배경도 자연스럽게 보정해줘."
        self.assertEqual(parse_image_command(natural_edit, source_image_available=True), ("pose", natural_edit))
        pose_feedback = [
            "아니 얼굴이 기울었잖아. 똑바로 보는 것으로 바꿔달라구",
            "이제 얼굴 모양은 됐고 정면을 바라보는 모습으로 바꿔달라고. 지금 고개가 약간 기울었잖아",
            "얼굴을 증명사진처럼 정면을 쳐다보게 바꿔달라고.",
            "고개가 약간 기울어져 있으니 똑바로 바라보게 수정해줘.",
        ]
        for feedback in pose_feedback:
            self.assertEqual(parse_image_command(feedback, source_image_available=True), ("pose", feedback))
        self.assertEqual(
            parse_image_command("사진 다시 보내줘야지", source_image_available=True),
            ("resend", "사진 다시 보내줘야지"),
        )
        self.assertIsNone(parse_image_command(natural_edit))
        self.assertIsNone(parse_image_command("이 사진을 설명해줘", source_image_available=True))
        self.assertIsNone(parse_image_command("일반 대화"))
        self.assertTrue(prefers_original_source("원래 얼굴로 되돌려줘"))
        self.assertFalse(prefers_original_source("배경을 조금 더 밝게 수정해줘"))
        with self.assertRaisesRegex(ValueError, "프롬프트"):
            parse_image_command("/image")

    def test_source_image_is_normalized_to_model_size(self) -> None:
        image = image_app._source_image(base64.b64encode(png_bytes()).decode())

        self.assertEqual(image.size, (512, 512))
        self.assertEqual(image.mode, "RGB")

    def test_image_endpoint_returns_png_metadata(self) -> None:
        client = TestClient(image_app.app)
        with patch.object(image_app.engine, "render", return_value=(png_bytes(), 42, "generate")):
            response = client.post("/v1/image", json={"prompt": "red square"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertEqual(response.headers["x-image-seed"], "42")
        self.assertEqual(response.headers["x-image-mode"], "generate")

    def test_prompt_optimizer_falls_back_when_qwen_is_unavailable(self) -> None:
        with patch("image_service.app.httpx.post", side_effect=OSError("offline")):
            self.assertEqual(image_app._image_prompt("원문 프롬프트"), "원문 프롬프트")

    def test_edit_prompt_preserves_identity_and_treats_complaints_as_corrections(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"choices": [{"message": {"content": "corrected portrait"}}]}

        with patch("image_service.app.httpx.post", return_value=response) as post:
            prompt = image_app._image_prompt("외모가 너무 달라졌어", editing=True)

        instruction = post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertEqual(prompt, "corrected portrait")
        self.assertIn("Preserve the same person's identity", instruction)
        self.assertIn("traits to remove", instruction)


if __name__ == "__main__":
    unittest.main()