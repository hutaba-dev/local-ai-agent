"""Single-process local SD-Turbo text-to-image and image-to-image service."""

from __future__ import annotations

import base64
import io
import os
import secrets
from threading import Lock

import httpx
import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from PIL import Image, ImageOps
from pydantic import BaseModel, Field


MODEL_ID = os.getenv("IMAGE_MODEL_ID", "stabilityai/sd-turbo")
MODEL_CACHE = os.getenv("IMAGE_MODEL_CACHE", "/srv/local-ai-agent/huggingface")
QWEN_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
QWEN_MODEL = os.getenv("OPENAI_MODEL", "qwen3.8-27b")
IMAGE_SIZE = 512


class ImageRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=2_000)
    source_image_base64: str | None = None
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)
    strength: float = Field(default=0.5, ge=0.25, le=0.95)


class ImageEngine:
    def __init__(self) -> None:
        self._text_pipeline: StableDiffusionPipeline | None = None
        self._edit_pipeline: StableDiffusionImg2ImgPipeline | None = None
        self._lock = Lock()

    def _load(self) -> StableDiffusionPipeline:
        if self._text_pipeline is None:
            pipeline = StableDiffusionPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
                variant="fp16",
                cache_dir=MODEL_CACHE,
            )
            pipeline.set_progress_bar_config(disable=True)
            pipeline.to("cuda")
            self._text_pipeline = pipeline
        return self._text_pipeline

    def render(self, request: ImageRequest) -> tuple[bytes, int, str]:
        with self._lock:
            pipeline = self._load()
            prompt = _image_prompt(request.prompt, editing=bool(request.source_image_base64))
            seed = request.seed if request.seed is not None else secrets.randbits(63)
            generator = torch.Generator(device="cuda").manual_seed(seed)
            if request.source_image_base64:
                source = _source_image(request.source_image_base64)
                if self._edit_pipeline is None:
                    self._edit_pipeline = StableDiffusionImg2ImgPipeline(**pipeline.components)
                    self._edit_pipeline.set_progress_bar_config(disable=True)
                output = self._edit_pipeline(
                    prompt=prompt,
                    image=source,
                    generator=generator,
                    num_inference_steps=4,
                    strength=request.strength,
                    guidance_scale=0.0,
                ).images[0]
                mode = "edit"
            else:
                output = pipeline(
                    prompt=prompt,
                    generator=generator,
                    num_inference_steps=4,
                    guidance_scale=0.0,
                    height=IMAGE_SIZE,
                    width=IMAGE_SIZE,
                ).images[0]
                mode = "generate"
            buffer = io.BytesIO()
            output.save(buffer, format="PNG", optimize=True)
            return buffer.getvalue(), seed, mode


def _source_image(encoded: str) -> Image.Image:
    try:
        content = base64.b64decode(encoded, validate=True)
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as error:
        raise ValueError("source image is invalid") from error
    return ImageOps.fit(image, (IMAGE_SIZE, IMAGE_SIZE), method=Image.Resampling.LANCZOS)


def _image_prompt(prompt: str, editing: bool = False) -> str:
    instruction = (
        "Rewrite the user's image request as one concise English diffusion-model prompt. "
        "Preserve every requested subject, action, composition, camera/view, color, style, background, and must-have object. "
        "Prioritize subject and action correctness, anatomy, and face quality before style fidelity. "
        "For people performing an action, use a readable full-body composition when needed, show the complete core object, "
        "make limb placement and balance physically coherent, keep the face unobscured unless requested, and use a clear silhouette. "
        "When an attractive or prominent face is requested, require clear facial structure, symmetrical aligned eyes, a natural "
        "defined nose and mouth, and clean facial line work. Pair requested styles with polished rendering and coherent anatomy. "
        "Include constraints against distorted faces, broken anatomy, extra limbs, melted hands, disfigured feet, unclear poses, "
        "missing core objects, and floating bodies. Return only the prompt."
    )
    if editing:
        instruction += (
            " This is an edit of an existing photo. Preserve the same person's identity, age, ethnicity, "
            "facial proportions, hair, clothing, and framing unless the user explicitly requests a change. "
            "Explicitly state what to keep, what to change, and which quality issue to fix. "
            "Interpret complaints such as 'became X', 'too X', or 'not X' as traits to remove, never as desired traits. "
            "Keep the final prompt under 45 English words so no requested edit or identity constraint is truncated."
        )
    try:
        response = httpx.post(
            f"{QWEN_BASE_URL}/chat/completions",
            json={
                "model": QWEN_MODEL,
                "messages": [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 80 if editing else 180,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=60,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        if isinstance(content, str) and content.strip():
            return content.strip()[:2_000]
    except (OSError, httpx.HTTPError, KeyError, IndexError, TypeError, ValueError):
        pass
    return prompt


engine = ImageEngine()
app = FastAPI(title="Local Image Generator", docs_url=None, redoc_url=None)


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "model": MODEL_ID, "loaded": engine._text_pipeline is not None}


@app.post("/v1/image")
def image(request: ImageRequest) -> Response:
    if not torch.cuda.is_available():
        raise HTTPException(status_code=503, detail="CUDA is unavailable")
    try:
        content, seed, mode = engine.render(request)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return Response(
        content,
        media_type="image/png",
        headers={"X-Image-Seed": str(seed), "X-Image-Mode": mode},
    )