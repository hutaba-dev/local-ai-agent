import base64
import json
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from PIL import Image

from image_service import app as image_app
from runtime import image_client
from runtime.image_client import build_image_edit_plan, parse_image_command, prefers_original_source


def png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (32, 24), "red").save(buffer, format="PNG")
    return buffer.getvalue()


class ImageServiceTests(unittest.TestCase):
    def test_multi_intent_edit_plan_preserves_every_requested_modification(self) -> None:
        cases = {
            "얼굴을 좀 더 잘생기게 하고 정면을 바라보게 해줘": (
                "face_orientation", "appearance_refinement",
            ),
            "머리를 길게 만들고 옷을 검정색으로 바꿔줘": ("hair_edit", "clothing_edit"),
            "웃게 하고 얼굴을 왼쪽으로 돌려줘": ("expression_edit", "face_orientation"),
            "배경을 바다로 바꾸고 옷은 흰색으로 하고 얼굴은 그대로 둬": (
                "clothing_edit", "background_edit",
            ),
            "얼굴만 정면으로 돌려줘": ("face_orientation",),
        }
        with patch("runtime.image_client.httpx.post", side_effect=OSError("offline")):
            for request, expected in cases.items():
                with self.subTest(request=request):
                    plan = build_image_edit_plan(request)
                    self.assertEqual(tuple(edit.type for edit in plan.edits), expected)
                    self.assertTrue(plan.preserve_identity)

            handsome_plan = build_image_edit_plan(next(iter(cases)))
        self.assertIn("balanced proportions", handsome_plan.edits[1].instruction)
        self.assertEqual(handsome_plan.edits[0].capability, "pose_correction")

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
        with patch("runtime.image_client.infer_image_intent", return_value="image"):
            natural_generation = "예쁜 여자가 물을 마시는 그림을 그려줘. 일본 애니메이션 스타일로."
            self.assertEqual(parse_image_command(natural_generation), ("image", natural_generation))
            english_generation = "Create an image of a glass tower at sunset."
            self.assertEqual(parse_image_command(english_generation), ("image", english_generation))
        with patch("runtime.image_client.infer_image_intent", return_value="chat"):
            self.assertIsNone(parse_image_command("일본 애니메이션 그림의 역사를 설명해줘"))
            self.assertIsNone(parse_image_command("이미지 생성 방법을 알려줘"))
            self.assertIsNone(parse_image_command("일반 대화"))
        self.assertTrue(prefers_original_source("원래 얼굴로 되돌려줘"))
        self.assertFalse(prefers_original_source("배경을 조금 더 밝게 수정해줘"))
        with self.assertRaisesRegex(ValueError, "프롬프트"):
            parse_image_command("/image")

    def test_continuation_intent_distinguishes_actions_from_conversation(self) -> None:
        cases = {
            "더 예쁘고 사실적으로... 고화질로": "edit",
            "얼굴이 망가졌네. 그림을 제대로 다시 그려줘.": "regenerate",
            "스케이트보드를 안 타잖아. 이상한 그림이야.": "regenerate",
            "정면을 보게 해줘": "pose",
            "고개가 기울어져 있으니 똑바로 바라보게 수정해줘": "pose",
            "결과 다시 보여줘": "resend",
            "사진 다시 보내줘야지": "resend",
            "앞으로는 고화질이라고 하면 이런식이어야해.": "chat",
            "고화질이 무슨 뜻이야?": "chat",
            "이 사진을 설명해줘": "chat",
        }
        for message, intent in cases.items():
            with self.subTest(message=message), patch(
                "runtime.image_client.infer_image_intent", return_value=intent
            ) as infer:
                expected = None if intent == "chat" else (intent, message)
                self.assertEqual(parse_image_command(message, source_image_available=True), expected)
                infer.assert_called_once_with(message, True)

    def test_continuation_intent_classifier_is_conservative(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"choices": [{"message": {"content": "CHAT"}}]}
        with patch("runtime.image_client.httpx.post", return_value=response) as post:
            intent = image_client.infer_image_intent("앞으로는 고화질이라고 하면 이런식이어야해.", True)
        self.assertEqual(intent, "chat")
        instruction = post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertIn("speech act rather than matching keywords", instruction)
        self.assertIn("distorted face", instruction)
        self.assertIn("clearly localized correction", instruction)
        self.assertIn("If the intent is ambiguous, choose chat", instruction)

        with patch("runtime.image_client.httpx.post", side_effect=OSError("offline")):
            self.assertEqual(image_client.infer_image_intent("더 예쁘고 사실적으로", True), "chat")
            self.assertEqual(image_client.infer_image_intent("오늘 날씨 어때?"), "chat")

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

    def test_edit_prompt_optimizer_uses_clip_safe_output_budget(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"choices": [{"message": {"content": "concise edit prompt"}}]}
        with patch("image_service.app.httpx.post", return_value=response) as post:
            prompt = image_app._image_prompt("apply two edits", editing=True)

        self.assertEqual(prompt, "concise edit prompt")
        self.assertEqual(post.call_args.kwargs["json"]["max_tokens"], 80)
        self.assertIn("under 45 English words", post.call_args.kwargs["json"]["messages"][0]["content"])

    def test_structured_prompt_builder_prioritizes_action_anatomy_and_face(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"choices": [{"message": {"content": json.dumps({
            "prompt": "full-body anime skateboard action",
            "subject": "student",
            "action": "riding a skateboard",
            "style": "anime",
            "face_priority": "high",
            "anatomy_priority": "high",
            "must_have_object": "skateboard",
        })}}]}

        with patch("runtime.image_client.httpx.post", return_value=response) as post:
            plan = image_client.build_image_prompt("예쁜 여학생이 스케이트보드를 타는 애니메이션 그림")

        instruction = post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertEqual(plan.must_have_object, "skateboard")
        self.assertIn("subject/action correctness, anatomy, face quality", instruction)
        self.assertIn("full body", instruction)
        self.assertIn("distorted faces", instruction)

    def test_quality_gate_requires_multiple_major_failures_before_retry(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"choices": [{"message": {"content": json.dumps({
            "subject": True,
            "action": False,
            "face": False,
            "anatomy": True,
            "main_object": True,
            "style": True,
            "summary": "action and face failed",
        })}}]}

        with patch("runtime.image_client.httpx.post", return_value=response):
            quality = image_client.assess_image_quality("a skateboard rider", png_bytes())

        self.assertTrue(quality.checked)
        self.assertFalse(quality.passed)
        self.assertEqual(quality.failures, ("action", "face"))

    def test_action_style_profiles_and_feedback_labels_are_generic(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"choices": [{"message": {"content": json.dumps({
            "prompt": "readable cyclist",
            "subject": "person",
            "action": "cycling",
            "style": "watercolor",
            "face_priority": "normal",
            "anatomy_priority": "high",
            "must_have_object": "bicycle",
        })}}]}

        with patch("runtime.image_client.httpx.post", return_value=response) as post:
            plan = image_client.build_image_prompt("수채화 스타일로 자전거를 타는 사람")

        planner_input = post.call_args.kwargs["json"]["messages"][1]["content"]
        self.assertEqual(plan.action_profile, "cycling")
        self.assertEqual(plan.style_profile, "watercolor")
        self.assertIn("feet aligned with pedals", planner_input)
        self.assertIn("controlled translucent washes", planner_input)
        self.assertEqual(
            image_client.feedback_failure_labels("얼굴과 손이 이상하고 자전거가 없어"),
            ("face_failure", "anatomy_failure", "object_failure"),
        )
        self.assertTrue(image_client.is_explicit_visual_preference("앞으로 그림은 항상 깔끔한 애니 스타일을 선호해"))
        self.assertFalse(image_client.is_explicit_visual_preference("예쁜 애니 스타일로 그려줘"))
        self.assertTrue(image_client.is_quality_sensitive_request("예쁜 여학생 캐릭터", "generic", "anime_illustration"))
        self.assertFalse(image_client.is_quality_sensitive_request("파란 정육면체", "generic", "generic"))

    def test_visual_reference_analysis_extracts_cues_without_identity_copying(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"choices": [{"message": {"content": "three-quarter pose, clean linework"}}]}

        with patch("runtime.image_client.httpx.post", return_value=response) as post:
            cues = image_client.analyze_visual_references(
                "anime walking scene", (("reference", "image/png", png_bytes()),)
            )

        instruction = post.call_args.kwargs["json"]["messages"][0]["content"][0]["text"]
        self.assertEqual(cues, "three-quarter pose, clean linework")
        self.assertIn("Do not identify or reproduce", instruction)

    def test_image_quality_benchmark_covers_required_scene_types_and_metrics(self) -> None:
        benchmark = json.loads(
            (Path(__file__).parent / "fixtures" / "image_quality_benchmarks.json").read_text(encoding="utf-8")
        )
        fixtures = benchmark["cases"]

        self.assertEqual(len(fixtures), 8)
        self.assertEqual(set(benchmark["evaluation_metrics"]), {
            "face_attractiveness", "facial_coherence", "body_coherence", "pose_readability",
            "object_correctness", "style_fidelity", "prompt_quality", "retry_decision_quality",
        })
        self.assertEqual({fixture["id"] for fixture in fixtures}, {
            "attractive-student-riding-skateboard", "attractive-student-riding-bicycle",
            "handsome-man-walking", "anime-full-body-person", "anime-face-closeup",
            "photorealistic-portrait", "person-action-with-background", "aesthetic-character-without-reference",
        })
        for fixture in fixtures:
            self.assertEqual(set(fixture["expected"]), {"action", "core_object", "composition", "quality_priority"})

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