"""Client contract for the local image generation service."""

from __future__ import annotations

import base64
import json
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


@dataclass(frozen=True)
class ImagePromptPlan:
    prompt: str
    subject: str
    action: str
    style: str
    face_priority: str
    anatomy_priority: str
    must_have_object: str


@dataclass(frozen=True)
class ImageQualityAssessment:
    checked: bool
    passed: bool
    failures: tuple[str, ...] = ()
    summary: str = ""


def build_image_prompt(
    request: str,
    *,
    editing: bool = False,
    original_request: str = "",
    quality_feedback: str = "",
    simplify_composition: bool = False,
) -> ImagePromptPlan:
    instruction = """Build a high-quality diffusion prompt from the user's image request.
Return one JSON object with string fields: prompt, subject, action, style, face_priority, anatomy_priority, must_have_object.
The prompt must be concise English and prioritize, in order: subject/action correctness, anatomy, face quality, then style fidelity.
Include an appropriate composition and camera/view, background, all must-have elements, and explicit failure constraints.
For a person performing an action, show the full body when needed, keep the core object fully visible, make limb placement and balance physically readable, keep the face unobscured unless requested, and use a clear silhouette.
For requested attractive or prominent people, require clear facial structure, symmetrical aligned eyes, a defined natural nose and mouth, and clean facial line work.
Combine a requested style with polished rendering, coherent anatomy, and character-art quality. Avoid distorted faces, broken anatomy, extra limbs, melted hands, disfigured feet, unclear poses, missing core objects, and floating bodies.
Do not imitate a living artist. Generic named media styles such as Japanese anime are allowed."""
    if editing:
        instruction += """
This is a localized edit. The prompt must explicitly say what to keep, what to change, and which quality issue to fix. Preserve successful identity, pose, composition, core objects, background, and style unless the user explicitly changes them."""
    else:
        instruction += """
This is generation from scratch. Do not refer to or depend on a previous image."""
    if simplify_composition:
        instruction += """
This is a retry after a failed result. Simplify the composition and camera angle while preserving the requested subject, action, core object, and style. Prefer a readable side or three-quarter view over a busy action shot."""
    user_content = (
        f"Original request: {original_request or request}\n"
        f"Current request or feedback: {request}\n"
        f"Known quality failures: {quality_feedback or 'none'}"
    )
    try:
        response = httpx.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            json={
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.1,
                "max_tokens": 500,
                "response_format": {"type": "json_object"},
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = json.loads(response.json()["choices"][0]["message"]["content"])
        prompt = str(payload.get("prompt", "")).strip()
        if prompt:
            return ImagePromptPlan(
                prompt=prompt[:2_000],
                subject=str(payload.get("subject", "unspecified"))[:160],
                action=str(payload.get("action", "none"))[:160],
                style=str(payload.get("style", "unspecified"))[:160],
                face_priority=str(payload.get("face_priority", "normal"))[:80],
                anatomy_priority=str(payload.get("anatomy_priority", "normal"))[:80],
                must_have_object=str(payload.get("must_have_object", "none"))[:160],
            )
    except (OSError, httpx.HTTPError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        pass
    fallback = original_request or request
    if quality_feedback and quality_feedback != request:
        fallback = f"{fallback}. Correct these failures: {quality_feedback}"
    fallback += (
        ". Clear readable composition and camera view. Show the complete subject and every core object. "
        "For people in action, show a readable full-body pose with coherent balance and limb placement. "
        "Clear symmetrical face, natural aligned eyes, defined nose and mouth, coherent anatomy, polished rendering. "
        "Avoid distorted faces, broken anatomy, extra limbs, melted hands, disfigured feet, unclear action, "
        "missing objects, or floating bodies."
    )
    return ImagePromptPlan(fallback[:2_000], "unspecified", "unspecified", "unspecified", "high", "high", "unspecified")


def assess_image_quality(request: str, image_content: bytes) -> ImageQualityAssessment:
    instruction = """Evaluate whether the generated image satisfies the request. Inspect exactly these criteria:
subject present, requested action clearly readable, face coherent when visible, anatomy acceptable, main object present and readable, and requested style roughly correct.
Return one JSON object with booleans subject, action, face, anatomy, main_object, style and a short summary string.
Use true for a criterion that is not applicable. Be strict about severe visible failures, but do not reject harmless artistic variation."""
    try:
        response = httpx.post(
            f"{OPENAI_BASE_URL}/chat/completions",
            json={
                "model": OPENAI_MODEL,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{instruction}\nUser request: {request}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64.b64encode(image_content).decode()}"}},
                    ],
                }],
                "temperature": 0,
                "max_tokens": 220,
                "response_format": {"type": "json_object"},
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=90,
        )
        response.raise_for_status()
        payload = json.loads(response.json()["choices"][0]["message"]["content"])
        criteria = ("subject", "action", "face", "anatomy", "main_object", "style")
        failures = tuple(criterion for criterion in criteria if payload.get(criterion) is False)
        return ImageQualityAssessment(True, len(failures) < 2, failures, str(payload.get("summary", ""))[:300])
    except (OSError, httpx.HTTPError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return ImageQualityAssessment(False, True, (), "quality gate unavailable")


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
- edit: a small, localized change where the existing subject, identity, composition, pose, or background should be preserved.
- regenerate: a request to try again from scratch because the result has a severe quality failure, such as a distorted face, broken anatomy, unreadable action, wrong composition, missing core object, major style mismatch, or increasing edit drift.
- pose: a present request involving the subject's head direction, gaze, or front-facing pose. This takes priority when combined with other image changes.
- resend: a present request to display or send the existing image again without changing it.
- chat: all ordinary conversation, including commentary, feedback, future preferences, explanations, questions, and acknowledgements.
Treat direct complaints such as "the face is broken", "this looks wrong", "what did you draw", or "the person is not riding it" as requests to fix a severe failure now and choose regenerate when a source image is available. Do not use regenerate for a clearly localized correction that should preserve a successful image. Editing, regeneration, pose correction, and resending require an available source image; otherwise choose chat. If the intent is ambiguous, choose chat."""
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
            allowed.update({"edit", "regenerate", "pose", "resend"})
        return label if label in allowed else "chat"
    except (OSError, httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
        return "chat"
