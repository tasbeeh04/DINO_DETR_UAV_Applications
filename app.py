"""Flask entrypoint for the dual-model CV demo.

Routes:
  GET  /              -> index page
  GET  /api/health    -> liveness + which models are loaded
  POST /api/predict   -> multipart/form-data image, returns URLs of two outputs
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, url_for

from cv_models import yolo_model, dino_model

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "static" / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_MIME = {"image/jpeg", "image/png"}
ALLOWED_EXT = {".jpg", ".jpeg", ".png"}
MAX_BYTES = 10 * 1024 * 1024  # 10 MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "models": ["yolo", "dino"]})


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "missing 'image' field"}), 400

    f = request.files["image"]
    if not f.filename:
        return jsonify({"error": "empty filename"}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXT or f.mimetype not in ALLOWED_MIME:
        return jsonify({"error": f"unsupported file type: {f.mimetype} ({ext})"}), 400

    job_id = uuid.uuid4().hex
    in_path = UPLOAD_DIR / f"{job_id}{ext}"
    f.save(in_path)

    yolo_out = OUTPUT_DIR / f"{job_id}_yolo.jpg"
    dino_out = OUTPUT_DIR / f"{job_id}_dino.jpg"

    try:
        yolo_model.predict(in_path, yolo_out)
    except Exception as e:
        app.logger.exception("YOLO inference failed")
        return jsonify({"error": f"yolo failed: {e}"}), 500

    try:
        dino_model.predict(in_path, dino_out)
    except Exception as e:
        app.logger.exception("DINO inference failed")
        return jsonify({"error": f"dino failed: {e}"}), 500

    return jsonify({
        "id": job_id,
        "yolo_url": url_for("static", filename=f"outputs/{yolo_out.name}"),
        "dino_url": url_for("static", filename=f"outputs/{dino_out.name}"),
    })


if __name__ == "__main__":
    import os
    from waitress import serve

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))

    print("Loading models...")
    yolo_model.load()
    dino_model.load()
    print(f"Models loaded. Serving on http://{host}:{port}")
    serve(app, host=host, port=port, threads=2)
