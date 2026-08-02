"""
Oil tank detection using YOLOv8 ONNX format (via OpenCV DNN) + shadow-based volume estimation.
"""
import base64
import pathlib
import warnings
import cv2
import numpy as np

warnings.filterwarnings("ignore")

from app.shadows_estimator import MultiTank

# ──────────────────────────── Paths ────────────────────────────────────

BASE_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
WEIGHTS_PATH = PROJECT_ROOT / "best_oil_tanks_3class.onnx"

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
    """Load the YOLOv8 ONNX model using OpenCV DNN."""
    path = weights_path or WEIGHTS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Model weights not found at {path}.")
    
    net = cv2.dnn.readNetFromONNX(str(path))
    print(f"YOLOv8 ONNX model loaded via OpenCV from {path}")
    return net

# ──────────────────────────── Detection ────────────────────────────────

def detect(net, image_array, conf=0.25):
    """Run ONNX detection using OpenCV DNN."""
    img_h, img_w = image_array.shape[:2]
    
    # Preprocess: YOLOv8 expects 512x512, RGB, /255.0, BCHW
    # OpenCV's blobFromImage handles resizing, scaling, and BCHW transposition
    blob = cv2.dnn.blobFromImage(image_array, 1/255.0, (512, 512), swapRB=False, crop=False)
    net.setInput(blob)
    outputs = net.forward()
    
    # outputs shape: (1, 7, 5376) -> transpose to (5376, 7)
    preds = outputs[0].T
    
    boxes = []
    scores = []
    class_ids = []
    
    x_factor = img_w / 512.0
    y_factor = img_h / 512.0
    
    for row in preds:
        classes_scores = row[4:]
        max_score = np.amax(classes_scores)
        if max_score >= conf:
            class_id = np.argmax(classes_scores)
            
            x, y, w, h = row[0], row[1], row[2], row[3]
            
            left = int((x - w / 2) * x_factor)
            top = int((y - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            
            boxes.append([left, top, width, height])
            scores.append(float(max_score))
            class_ids.append(int(class_id))
            
    # NMS
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf, 0.45)
    
    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            box = boxes[i]
            x1, y1 = box[0], box[1]
            x2, y2 = box[0] + box[2], box[1] + box[3]
            cls_id = class_ids[i]
            
            detections.append({
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": scores[i],
                "class_id": cls_id,
                "class_name": CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
            })
            
    return detections

# ──────────────────────────── Volume Estimation ────────────────────────

def estimate_volumes(image_array, detections):
    h, w = image_array.shape[:2]

    fht_indices = [i for i, d in enumerate(detections) if d["class_id"] == 0]
    fht_bboxes = []
    for i in fht_indices:
        x1, y1, x2, y2 = detections[i]["bbox_xyxy"]
        fht_bboxes.append([
            int(round(y1)), int(round(x1)),
            int(round(y2)), int(round(x2)),
        ])

    image_float = image_array.astype(np.float32) / 255.0
    multi_tank = MultiTank(fht_bboxes, image_float)
    fht_volumes = multi_tank.get_volumes()

    from app.shadows_estimator import check_bb
    volumes = []
    fht_vol_idx = 0
    for i, det in enumerate(detections):
        if det["class_id"] != 0:
            volumes.append(-1.0)
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
    img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    for det, vol in zip(detections, volumes):
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        conf = det["confidence"]
        cls_id = det["class_id"]
        color = CLASS_COLORS_BGR.get(cls_id, (255, 255, 255))
        short_name = CLASS_SHORT.get(cls_id, "?")

        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        if cls_id == 0 and vol >= 0:
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
    detections = detect(model, image_array, conf=conf)
    volumes = estimate_volumes(image_array, detections)

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

    annotated_bgr = draw_outputs(image_array, detections, volumes)
    retval, buffer = cv2.imencode(file_suffix, annotated_bgr)
    encoded_img = base64.b64encode(buffer).decode("utf-8")

    return {
        "data": [fht_results],
        "all_detections": all_results,
        "encoded_img": encoded_img,
    }
