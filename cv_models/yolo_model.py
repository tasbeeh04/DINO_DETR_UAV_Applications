"""YOLOv11x wrapper.

Loads the Ultralytics model once and exposes a `predict(image_path, out_path)`
function that writes an annotated JPEG to `out_path`.
"""

from __future__ import annotations

from pathlib import Path

import cv2
from ultralytics import YOLO

WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "weights" / "best (1).pt"

_model: YOLO | None = None


def load() -> YOLO:
    global _model
    if _model is None:
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError(f"YOLO weights not found at {WEIGHTS_PATH}")
        _model = YOLO(str(WEIGHTS_PATH))
    return _model


def predict(image_path: str | Path, out_path: str | Path, conf: float = 0.25) -> str:
    """Run YOLO on `image_path` and save the annotated image to `out_path`."""
    model = load()
    results = model.predict(source=str(image_path), conf=conf, save=False, verbose=False)
    annotated_bgr = results[0].plot()
    out_path = str(out_path)
    cv2.imwrite(out_path, annotated_bgr)
    return out_path
