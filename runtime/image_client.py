"""Client contract for the local image generation service."""

from __future__ import annotations

import base64
import os
import re
from dataclasses import dataclass

import httpx


IMAGE_API_URL = os.getenv("IMAGE_API_URL", "http://127.0.0.1:8001").rstrip("/")
POSE_API_URL = os.getenv("POSE_API_URL", "http://127.0.0.1:8002").rstrip("/")
IMAGE_WORKER_URL = os.getenv("IMAGE_WORKER_URL", "").rstrip("/")
IMAGE_WORKER_TOKEN = os.getenv("IMAGE_WORKER_TOKEN", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "qwen3.8-27b")
ORIGINAL_RESTORE_PATTERN = re.compile(
    r"원래|되돌|복원|처음\s*(?:사진|이미지|얼굴|모습)|"
    r"\b(?:original|restore|revert|reset)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GeneratedImage:
    content: bytes
    seed: int
    mode: str
    filename: str


def create_image(prompt: str, source_image: bytes | None = None, strength: float = 0.5) -> GeneratedImage:
    payload: dict[str, object] = {"prompt": prompt, "strength": strength}
    if source_image is not None:
        payload["source_image_base64"] = base64.b64encode(source_image).decode()
    if IMAGE_WORKER_URL:
        capability = "image.edit" if source_image is not None else "image.generate"
        response = _worker_post(capability, payload)
    else:
        response = httpx.post(f"{IMAGE_API_URL}/v1/image", json=payload, timeout=300)
    response.raise_for_status()
    seed = int(response.headers["X-Image-Seed"])
    mode = response.headers["X-Image-Mode"]
    return GeneratedImage(response.content, seed, mode, f"{mode}-{seed}.png")


def correct_portrait_pose(source_image: bytes) -> GeneratedImage:
    payload = {"source_image_base64": base64.b64encode(source_image).decode()}
    if IMAGE_WORKER_URL:
        response = _worker_post("portrait.frontalize", payload)
    else:
        response = httpx.post(f"{POSE_API_URL}/v1/frontalize", json=payload, timeout=300)
    response.raise_for_status()
    return GeneratedImage(response.content, 0, "pose", "pose-corrected.png")


def _worker_post(capability: str, payload: dict[str, object]) -> httpx.Response:
    if not IMAGE_WORKER_TOKEN:
        raise RuntimeError("IMAGE_WORKER_TOKEN is required when IMAGE_WORKER_URL is configured")
    return httpx.post(
        f"{IMAGE_WORKER_URL}/v1/tasks/{capability}",
        json=payload,
        headers={"Authorization": f"Bearer {IMAGE_WORKER_TOKEN}"},
        timeout=300,
    )


def parse_image_command(message: str, source_image_available: bool = False) -> tuple[str, str] | None:
    stripped_message = message.strip()
    command, separator, prompt = stripped_message.partition(" ")
    if command not in {"/image", "/edit"}:
        intent = infer_image_intent(stripped_message, source_image_available)
        return None if intent == "chat" else (intent, stripped_message)
    if not separator or not prompt.strip():
        raise ValueError(f"{command} 뒤에 프롬프트를 입력하세요.")
    return command[1:], prompt.strip()


def prefers_original_source(message: str) -> bool:
    return bool(ORIGINAL_RESTORE_PATTERN.search(message))


def infer_image_intent(message: str, source_image_available: bool = False) -> str:
    instruction = """Determine the user's present intent in a conversation that can generate and edit images.
Decide from the meaning, discourse context, and speech act rather than matching keywords.
Return exactly one lowercase label:
- image: a present request to create or draw a new image from a description.
- edit: a present request to alter, improve, restyle, add, remove, or regenerate the image.
- pose: a present request involving the subject's head direction, gaze, or front-facing pose. This takes priority when combined with other image changes.
- resend: a present request to display or send the existing image again without changing it.
- chat: all ordinary conversation, including commentary, feedback, future preferences, explanations, questions, and acknowledgements.
Only choose an image action when the user is asking for that action to be performed now. Editing, pose correction, and resending require an available source image; otherwise choose chat. If the intent is ambiguous, choose chat."""
    try:
        response = httpx.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            json={
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": instruction},
                    {
                        "role": "user",
                        "content": f"Source image available: {'yes' if source_image_available else 'no'}\nUser message: {message}",
                    },
                ],
                "temperature": 0,
                "max_tokens": 16,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=30,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        label = str(content).strip().lower().rstrip(".")
        allowed = {"image", "chat"}
        if source_image_available:
            allowed.update({"edit", "pose", "resend"})
        return label if label in allowed else "chat"
    except (OSError, httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
        return "chat"