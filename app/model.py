"""
Oil tank detection using YOLOv8 (Ultralytics) + shadow-based volume estimation.

Replaces the old TensorFlow/Keras YOLOv3 pipeline with our locally-trained
YOLOv8n model for floating-head tank detection.
"""

import base64
import pathlib
import warnings

warnings.filterwarnings("ignore")

import cv2
import numpy as np
from ultralytics import YOLO

from app.shadows_estimator import MultiTank

# ──────────────────────────── Paths ────────────────────────────────────

BASE_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
WEIGHTS_PATH = PROJECT_ROOT / "best_oil_tanks_3class.pt"

CLASS_NAMES = {
    0: "Floating Head Tank",
    1: "Fixed Roof Tank",
    2: "Tank Cluster",
}

CLASS_COLORS_BGR = {
    0: (0, 255, 0),    # Green for FHT
    1: (255, 165, 0),  # Orange for Fixed Roof
    2: (0, 0, 255),    # Red for Tank Cluster
}

CLASS_SHORT = {
    0: "FHT",
    1: "FRT",
    2: "TC",
}


# ──────────────────────────── Model Loading ────────────────────────────

def load_model(weights_path=None):
    """Load the trained YOLOv8 model."""
    path = weights_path or WEIGHTS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Model weights not found at {path}. "
            f"Run train_local.py first to train the model."
        )
    model = YOLO(str(path))
    print(f"YOLOv8 model loaded from {path}")
    return model


# ──────────────────────────── Detection ────────────────────────────────

def detect(model, image_array, conf=0.25):
    """
    Run YOLOv8 detection on an image.

    Args:
        model: Ultralytics YOLO model
        image_array: numpy array (H, W, 3) in RGB, uint8
        conf: confidence threshold

    Returns:
        List of dicts with keys: bbox (xyxy), confidence, class_name
    """
    results = model.predict(image_array, conf=conf, verbose=False)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls_id = int(box.cls[0])
        detections.append({
            "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
            "confidence": float(box.conf[0]),
            "class_id": cls_id,
            "class_name": CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
        })

    return detections


# ──────────────────────────── Volume Estimation ────────────────────────

def estimate_volumes(image_array, detections):
    """
    Run shadow extraction on detected tanks to estimate fill volumes.

    Args:
        image_array: numpy array (H, W, 3) in RGB, uint8
        detections: list of detection dicts from detect()

    Returns:
        List of volume estimates (float, 0-1) matching detections order
    """
    h, w = image_array.shape[:2]

    # Only run shadow extraction on Floating Head Tanks (class_id == 0)
    fht_indices = [i for i, d in enumerate(detections) if d["class_id"] == 0]
    fht_bboxes = []
    for i in fht_indices:
        x1, y1, x2, y2 = detections[i]["bbox_xyxy"]
        fht_bboxes.append([
            int(round(y1)),
            int(round(x1)),
            int(round(y2)),
            int(round(x2)),
        ])

    # Shadow estimator expects float [0, 1] image
    image_float = image_array.astype(np.float32) / 255.0

    multi_tank = MultiTank(fht_bboxes, image_float)
    fht_volumes = multi_tank.get_volumes()

    # Map volumes back: FHTs get shadow volume, others get -1 (not applicable)
    from app.shadows_estimator import check_bb
    volumes = []
    fht_vol_idx = 0
    for i, det in enumerate(detections):
        if det["class_id"] != 0:
            volumes.append(-1.0)  # Not a FHT, no volume estimate
        else:
            x1, y1, x2, y2 = det["bbox_xyxy"]
            bb = [int(round(y1)), int(round(x1)), int(round(y2)), int(round(x2))]
            if check_bb(bb, image_array.shape) and fht_vol_idx < len(fht_volumes):
                volumes.append(fht_volumes[fht_vol_idx])
                fht_vol_idx += 1
            else:
                volumes.append(0.0)

    return volumes


# ──────────────────────────── Drawing ──────────────────────────────────

def draw_outputs(image_array, detections, volumes):
    """
    Draw bounding boxes with confidence and volume labels.

    Args:
        image_array: numpy array (H, W, 3) in RGB, uint8
        detections: list of detection dicts
        volumes: list of volume estimates

    Returns:
        Annotated image as numpy array (BGR for OpenCV compatibility)
    """
    img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    for det, vol in zip(detections, volumes):
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        conf = det["confidence"]
        cls_id = det["class_id"]
        color = CLASS_COLORS_BGR.get(cls_id, (255, 255, 255))
        short_name = CLASS_SHORT.get(cls_id, "?")

        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        if cls_id == 0 and vol >= 0:  # FHT with volume
            label = f"{short_name} {conf:.2f} V:{vol:.2f}"
        else:
            label = f"{short_name} {conf:.2f}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        img = cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        img = cv2.putText(
            img, label, (x1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1,
        )

    return img


# ──────────────────────────── Full Pipeline ────────────────────────────

def make_prediction(model, image_array, file_suffix=".jpg", conf=0.25):
    """
    Full pipeline: detect → estimate volumes → draw → encode.

    Args:
        model: Ultralytics YOLO model
        image_array: numpy array (H, W, 3) in RGB, uint8
        file_suffix: image format for encoding (e.g. '.jpg', '.png')
        conf: confidence threshold for detection

    Returns:
        dict with keys:
            - data: list of detection results with volumes
            - encoded_img: base64-encoded annotated image
    """
    # 1. Detect tanks
    detections = detect(model, image_array, conf=conf)

    # 2. Estimate volumes via shadow extraction
    volumes = estimate_volumes(image_array, detections)

    # 3. Build result data — group by class for the API response
    fht_results = []
    all_results = []
    for det, vol in zip(detections, volumes):
        x1, y1, x2, y2 = det["bbox_xyxy"]
        entry = {
            "confidence": f"{det['confidence']:.4f}",
            "volumes": f"{vol:.4f}" if vol >= 0 else "N/A",
            "file_id": det["class_name"],
            "class_id": det["class_id"],
            "bbox": f"{x1:.4f} {y1:.4f} {x2:.4f} {y2:.4f}",
        }
        all_results.append(entry)
        if det["class_id"] == 0 and vol >= 0:
            fht_results.append(entry)

    # 4. Draw annotated image
    annotated_bgr = draw_outputs(image_array, detections, volumes)

    # 5. Encode to base64
    retval, buffer = cv2.imencode(file_suffix, annotated_bgr)
    encoded_img = base64.b64encode(buffer).decode("utf-8")

    return {
        "data": [fht_results],       # FHT-only results for volume/barrel calc
        "all_detections": all_results, # all 3-class detections for display
        "encoded_img": encoded_img,
    }
