"""FastAPI wrapper around LivePortrait source-pose retargeting."""

from __future__ import annotations

import base64
import io
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel, Field
import torch


LIVEPORTRAIT_ROOT = Path(os.getenv("LIVEPORTRAIT_ROOT", "/srv/local-ai-agent/liveportrait"))
sys.path.insert(0, str(LIVEPORTRAIT_ROOT))

from src.config.argument_config import ArgumentConfig  # noqa: E402
from src.config.crop_config import CropConfig  # noqa: E402
from src.config.inference_config import InferenceConfig  # noqa: E402
from src.gradio_pipeline import GradioPipeline  # noqa: E402


class PoseRequest(BaseModel):
    source_image_base64: str = Field(min_length=1)


def _partial_fields(target_class, values: dict[str, object]):
    return target_class(**{key: value for key, value in values.items() if hasattr(target_class, key)})


class PoseEngine:
    def __init__(self) -> None:
        self._pipeline: GradioPipeline | None = None
        self._lock = Lock()

    def _load(self) -> GradioPipeline:
        if self._pipeline is None:
            args = ArgumentConfig()
            self._pipeline = GradioPipeline(
                inference_cfg=_partial_fields(InferenceConfig, args.__dict__),
                crop_cfg=_partial_fields(CropConfig, args.__dict__),
                args=args,
            )
        return self._pipeline

    def frontalize(self, content: bytes) -> tuple[bytes, float, float, float]:
        try:
            with Image.open(io.BytesIO(content)) as source:
                source.verify()
        except Exception as error:
            raise ValueError("source image is invalid") from error

        with TemporaryDirectory(prefix="local-ai-pose-") as directory:
            source_path = Path(directory) / "source.png"
            source_path.write_bytes(content)
            with self._lock:
                pipeline = self._load()
                eye_ratio, lip_ratio = pipeline.init_retargeting_image(2.5, 0.0, 0.0, str(source_path))
                prepared = pipeline.prepare_retargeting_image(
                    str(source_path), 0.0, 0.0, 0.0, 2.5, flag_do_crop=True
                )
                pose = prepared[4]
                pitch = float(pose["pitch"].item())
                yaw = float(pose["yaw"].item())
                roll = float(pose["roll"].item())
                _, result = pipeline.execute_image_retargeting(
                    eye_ratio, lip_ratio, -pitch, -yaw, -roll,
                    0.0, 0.0, 1.0,
                    0.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0,
                    str(source_path), 2.5, True, True,
                )
            torch.cuda.empty_cache()
        success, encoded = cv2.imencode(".png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        if not success:
            raise RuntimeError("could not encode pose-corrected image")
        return encoded.tobytes(), pitch, yaw, roll


engine = PoseEngine()
app = FastAPI(title="Local Portrait Pose Corrector", docs_url=None, redoc_url=None)


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "loaded": engine._pipeline is not None}


@app.post("/v1/frontalize")
def frontalize(request: PoseRequest) -> Response:
    try:
        content = base64.b64decode(request.source_image_base64, validate=True)
        result, pitch, yaw, roll = engine.frontalize(content)
    except (ValueError, RuntimeError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return Response(
        result,
        media_type="image/png",
        headers={
            "X-Source-Pitch": f"{pitch:.2f}",
            "X-Source-Yaw": f"{yaw:.2f}",
            "X-Source-Roll": f"{roll:.2f}",
        },
    )