"""DINO (IDEA-Research) wrapper.

Loads the IDEA-Research DINO detection model from
`weights/checkpoint_best_regular.pth`. The checkpoint embeds the original
training args (DINO_4scale, ResNet50 backbone, 10 VisDrone classes), so we
reuse them to rebuild the exact architecture.

The vendored repo lives in `vendor/dino_repo` and is added to sys.path on
first use.

Preprocessing here is *separate* from the YOLO pipeline:
  - resize shortest side to 800 (max long side 1333)
  - ToTensor + ImageNet mean/std normalization
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = ROOT / "weights" / "checkpoint_best_regular.pth"
DINO_REPO = ROOT / "vendor" / "dino_repo"

# VisDrone-2019 class names (10 classes, in the order the model emits)
CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]

# distinct color per class for drawing
CLASS_COLORS = [
    (255,  56,  56), ( 56, 255,  56), ( 56,  56, 255), (255, 255,  56),
    (255,  56, 255), ( 56, 255, 255), (255, 128,   0), (128,   0, 255),
    (  0, 200, 200), (200,   0, 100),
]

CONFIDENCE_THRESHOLD = 0.30

_model = None
_postprocessor = None
_device = torch.device("cpu")  # GPU not available in this environment


def _ensure_repo_on_path() -> None:
    """Add vendored DINO repo to sys.path so its `models`/`util` packages import."""
    p = str(DINO_REPO)
    if p not in sys.path:
        sys.path.insert(0, p)


def _build_preprocess() -> T.Compose:
    return T.Compose([
        T.Resize(800, max_size=1333),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


_preprocess = _build_preprocess()


def load():
    """Load the DINO model + postprocessor once, return (model, postprocessor)."""
    global _model, _postprocessor
    if _model is not None:
        return _model, _postprocessor

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"DINO weights not found at {WEIGHTS_PATH}")
    if not DINO_REPO.exists():
        raise FileNotFoundError(
            f"DINO source not found at {DINO_REPO}. "
            "Run: git clone --depth 1 https://github.com/IDEA-Research/DINO.git "
            f"{DINO_REPO}"
        )

    _ensure_repo_on_path()

    ckpt = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
    args = ckpt["args"]
    args.device = "cpu"  # override training-time 'cuda'
    args.distributed = False

    from models.registry import MODULE_BUILD_FUNCS  # noqa: E402
    import models.dino  # noqa: F401  (registers the 'dino' builder)

    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, _criterion, postprocessors = build_func(args)

    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"[dino] missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        print(f"[dino] unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")

    model.to(_device).eval()
    _model = model
    _postprocessor = postprocessors["bbox"]
    return _model, _postprocessor


def _draw(image: Image.Image, boxes, scores, labels, out_path: Path) -> None:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except OSError:
        font = ImageFont.load_default()

    for box, score, lbl in zip(boxes, scores, labels):
        if score < CONFIDENCE_THRESHOLD:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        cls = int(lbl)
        color = CLASS_COLORS[cls % len(CLASS_COLORS)]
        name = CLASS_NAMES[cls] if 0 <= cls < len(CLASS_NAMES) else str(cls)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{name} {score:.2f}"
        tb = draw.textbbox((x1, y1), text, font=font)
        draw.rectangle([tb[0], tb[1], tb[2] + 4, tb[3] + 2], fill=color)
        draw.text((x1 + 2, y1), text, fill="black", font=font)

    img.save(out_path, "JPEG", quality=90)


@torch.no_grad()
def predict(image_path: str | Path, out_path: str | Path) -> str:
    """Run DINO on `image_path` and save the annotated image to `out_path`."""
    model, postprocessor = load()

    pil = Image.open(str(image_path)).convert("RGB")
    tensor = _preprocess(pil).unsqueeze(0).to(_device)

    outputs = model(tensor)
    target_sizes = torch.tensor([[pil.height, pil.width]], device=_device)
    results = postprocessor(outputs, target_sizes)[0]

    out_path = Path(out_path)
    _draw(pil, results["boxes"].cpu(), results["scores"].cpu(), results["labels"].cpu(), out_path)
    return str(out_path)
