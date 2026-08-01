"""
FastAPI backend for oil tank volume estimation.

Endpoints:
    POST /prediction/  — upload a satellite image, returns detected tanks
                          with volume estimates and an annotated image.
"""

import io
import pathlib
import uuid

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from app.model import load_model, make_prediction, detect, estimate_volumes, draw_outputs

# ──────────────────────────── Setup ────────────────────────────────────

BASE_DIR = pathlib.Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "results"

# Load YOLOv8 model at startup
MODEL = load_model()

app = FastAPI(
    title="Oil Tank Volume Estimator",
    description="Detect floating-head oil tanks and estimate fill volumes from satellite images.",
    version="2.0.0",
)


# ──────────────────────────── Endpoints ────────────────────────────────

@app.post("/prediction/")
async def prediction_view(file: UploadFile = File(...), conf: float = 0.25):
    """
    Upload a satellite image. Returns:
    - results: list of detected tanks with confidence, volume, bbox
    - image_encoded: base64-encoded annotated image with detections
    """
    bytes_str = io.BytesIO(await file.read())
    try:
        img = Image.open(bytes_str).convert("RGB")
    except Exception:
        raise HTTPException(detail="Invalid image", status_code=400)

    image_array = np.array(img)
    fname = pathlib.Path(file.filename)
    fext = fname.suffix or ".jpg"

    # Run detection + shadow extraction + volume estimation
    predictions = make_prediction(MODEL, image_array, fext, conf=conf)

    # Save annotated image
    UPLOAD_DIR.mkdir(exist_ok=True)
    import base64
    decoded_img = base64.b64decode(predictions["encoded_img"])
    dest = UPLOAD_DIR / fname.name
    with open(dest, "wb") as f:
        f.write(decoded_img)

    return {
        "results": predictions["data"],
        "all_detections": predictions["all_detections"],
        "image_encoded": predictions["encoded_img"],
    }


@app.post("/img-echo/", response_class=FileResponse)
async def img_echo_view(file: UploadFile = File(...)):
    """Echo back an uploaded image (for testing)."""
    UPLOAD_DIR.mkdir(exist_ok=True)
    bytes_str = io.BytesIO(await file.read())
    try:
        img = Image.open(bytes_str)
    except Exception:
        raise HTTPException(detail="Invalid image", status_code=400)
    fname = pathlib.Path(file.filename)
    fext = fname.suffix or ".jpg"
    dest = UPLOAD_DIR / f"{uuid.uuid1()}{fext}"
    img.save(dest)
    return dest
