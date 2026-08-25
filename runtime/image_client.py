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
NATURAL_GENERATE_PATTERN = re.compile(
    r"(?:그림|이미지|일러스트|사진).{0,24}(?:그려|그려줘|그려주|만들어|만들어줘|생성해|생성해줘)|"
    r"(?:그려|그려줘|그려주|만들어|만들어줘|생성해|생성해줘).{0,24}(?:그림|이미지|일러스트|사진)|"
    r"\b(?:draw|generate|create|make)\b.{0,40}\b(?:image|picture|illustration|artwork|photo)\b|"
    r"\b(?:image|picture|illustration|artwork|photo)\b.{0,40}\b(?:draw|generate|create|make)\b",
    re.IGNORECASE,
)
NATURAL_EDIT_PATTERN = re.compile(
    r"수정|편집|보정|바꿔|바꾸|변경|지워|제거|없애|고쳐|"
    r"\b(?:edit|change|replace|remove|retouch|adjust|straighten|enhance)\b",
    re.IGNORECASE,
)
NATURAL_RESEND_PATTERN = re.compile(
    r"(?:사진|이미지|결과).{0,12}다시\s*(?:보내|보여|올려)|"
    r"다시\s*(?:사진|이미지|결과).{0,12}(?:보내|보여|올려)|"
    r"\b(?:resend|show|send)\b.{0,12}\b(?:image|photo|result)\b",
    re.IGNORECASE,
)
ORIGINAL_RESTORE_PATTERN = re.compile(
    r"원래|되돌|복원|처음\s*(?:사진|이미지|얼굴|모습)|"
    r"\b(?:original|restore|revert|reset)\b",
    re.IGNORECASE,
)
NATURAL_POSE_PATTERN = re.compile(
    r"고개.{0,16}(?:똑바로|정면)|(?:똑바로|정면).{0,16}고개|"
    r"(?:얼굴|사진).{0,24}(?:똑바로|정면|증명사진)|(?:똑바로|정면|증명사진).{0,24}(?:얼굴|사진)|"
    r"얼굴.{0,16}기울.{0,24}똑바로|기울.{0,24}똑바로\s*(?:보|바라보)|"
    r"바라보.{0,12}(?:정면|카메라)|(?:정면|카메라).{0,12}바라보|"
    r"\b(?:front-facing|frontal|straighten (?:the )?head|face the camera)\b",
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
        if source_image_available and NATURAL_RESEND_PATTERN.search(stripped_message):
            return "resend", stripped_message
        if source_image_available and NATURAL_POSE_PATTERN.search(stripped_message):
            return "pose", stripped_message
        if source_image_available and NATURAL_EDIT_PATTERN.search(stripped_message):
            return "edit", stripped_message
        if NATURAL_GENERATE_PATTERN.search(stripped_message):
            return "image", stripped_message
        return None
    if not separator or not prompt.strip():
        raise ValueError(f"{command} 뒤에 프롬프트를 입력하세요.")
    return command[1:], prompt.strip()


def prefers_original_source(message: str) -> bool:
    return bool(ORIGINAL_RESTORE_PATTERN.search(message))