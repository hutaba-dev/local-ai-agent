import base64
import unittest
from io import BytesIO
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from image_worker import app as worker_app


TOKEN = "test-worker-token"


def png_base64() -> str:
    buffer = BytesIO()
    Image.new("RGB", (16, 16), "blue").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


class ImageWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(worker_app.app)
        self.authorization = {"Authorization": f"Bearer {TOKEN}"}

    def test_rejects_missing_or_invalid_token(self) -> None:
        with patch.object(worker_app, "WORKER_TOKEN", TOKEN):
            self.assertEqual(self.client.get("/v1/capabilities").status_code, 401)
            response = self.client.get(
                "/v1/capabilities", headers={"Authorization": "Bearer wrong"}
            )
        self.assertEqual(response.status_code, 401)

    def test_reports_capabilities(self) -> None:
        with patch.object(worker_app, "WORKER_TOKEN", TOKEN):
            response = self.client.get("/v1/capabilities", headers=self.authorization)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["max_concurrency"], 1)
        self.assertIn("portrait.frontalize", response.json()["capabilities"])

    def test_routes_valid_generation_to_backend_manager(self) -> None:
        result = worker_app.BackendResult(
            b"png", "image/png", {"X-Image-Seed": "7", "X-Image-Mode": "generate"}
        )
        with patch.object(worker_app, "WORKER_TOKEN", TOKEN), patch.object(
            worker_app.backend_manager, "execute", return_value=result
        ) as execute:
            response = self.client.post(
                "/v1/tasks/image.generate",
                headers=self.authorization,
                json={"prompt": "blue square"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-image-seed"], "7")
        execute.assert_called_once_with(
            "image.generate", {"prompt": "blue square", "strength": 0.5}
        )

    def test_edit_requires_valid_source_image(self) -> None:
        with patch.object(worker_app, "WORKER_TOKEN", TOKEN), patch.object(
            worker_app.backend_manager, "execute", side_effect=RuntimeError("backend unavailable")
        ) as execute:
            missing = self.client.post(
                "/v1/tasks/image.edit", headers=self.authorization, json={"prompt": "edit"}
            )
            invalid = self.client.post(
                "/v1/tasks/image.edit",
                headers=self.authorization,
                json={"prompt": "edit", "source_image_base64": base64.b64encode(b"text").decode()},
            )
            valid = self.client.post(
                "/v1/tasks/image.edit",
                headers=self.authorization,
                json={"prompt": "edit", "source_image_base64": png_base64()},
            )
        self.assertEqual(missing.status_code, 422)
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(valid.status_code, 502)
        self.assertEqual(execute.call_count, 1)

    def test_edit_accepts_identity_preserving_strength(self) -> None:
        result = worker_app.BackendResult(
            b"png", "image/png", {"X-Image-Seed": "8", "X-Image-Mode": "edit"}
        )
        with patch.object(worker_app, "WORKER_TOKEN", TOKEN), patch.object(
            worker_app.backend_manager, "execute", return_value=result
        ) as execute:
            response = self.client.post(
                "/v1/tasks/image.edit",
                headers=self.authorization,
                json={"prompt": "subtle face edit", "source_image_base64": png_base64(), "strength": 0.25},
            )
        self.assertEqual(response.status_code, 200)
        execute.assert_called_once_with(
            "image.edit",
            {"prompt": "subtle face edit", "source_image_base64": png_base64(), "strength": 0.25},
        )

    def test_rejects_unknown_capability_without_backend_call(self) -> None:
        with patch.object(worker_app, "WORKER_TOKEN", TOKEN), patch.object(
            worker_app.backend_manager, "execute"
        ) as execute:
            response = self.client.post(
                "/v1/tasks/shell.execute", headers=self.authorization, json={"command": "id"}
            )
        self.assertEqual(response.status_code, 404)
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()