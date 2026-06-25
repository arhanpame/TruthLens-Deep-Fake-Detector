import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image

import model_setup
import helpers

app = FastAPI(title="DeepFake Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_state = {"model": None, "class_names": None}


def get_model():
    if _state["model"] is None:
        import tensorflow as tf

        with open(model_setup.paths["class_names.json"]) as f:
            _state["class_names"] = json.load(f)
        _state["model"] = tf.keras.models.load_model(
            model_setup.paths["deepfake_detector.keras"]
        )
    return _state["model"], _state["class_names"]


def _samples_dir():
    return os.path.join(model_setup.dir, "sample_images")


@app.get("/api/healthz")
def healthz():
    return {"status": "ok", "model_loaded": _state["model"] is not None}


@app.get("/api/samples")
def samples():
    folder = _samples_dir()
    items = []
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        truth = "AI-Generated" if fname.lower().startswith("ai") else "Real"
        items.append({"name": fname, "truth": truth, "url": f"api/samples/{fname}"})
    return {"samples": items}


@app.get("/api/samples/{name}")
def sample_image(name: str):
    folder = _samples_dir()
    safe = os.path.basename(name)
    path = os.path.join(folder, safe)
    if not os.path.isfile(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path)


def _predict_pil(image: Image.Image):
    try:
        model, class_names = get_model()
        t0 = time.time()
        label, scores = helpers.predict(model, image, tuple(class_names))
        elapsed = (time.time() - t0) * 1000.0
    except Exception:
        return JSONResponse(
            {"error": "Model inference failed. Please try again."},
            status_code=500,
        )
    return {
        "label": label,
        "scores": {k: float(v) for k, v in scores.items()},
        "confidence": float(scores[label]),
        "inference_ms": round(elapsed, 1),
        "class_names": list(class_names),
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return JSONResponse({"error": "Invalid image file"}, status_code=400)
    return _predict_pil(image)


@app.post("/api/predict-sample/{name}")
def predict_sample(name: str):
    folder = _samples_dir()
    safe = os.path.basename(name)
    path = os.path.join(folder, safe)
    if not os.path.isfile(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    image = Image.open(path).convert("RGB")
    return _predict_pil(image)
