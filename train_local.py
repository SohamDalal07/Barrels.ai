"""
Oil Tank Detection — Local YOLOv8 Training Script (3-Class)
===========================================================
Trains a YOLOv8n model to detect 3 classes of oil storage tanks:
  0: Floating Head Tank
  1: Fixed Roof Tank ("Tank")
  2: Tank Cluster

Using the `towardsentropy/oil-storage-tanks` Kaggle dataset.
Optimized for NVIDIA RTX 3050 Laptop GPU (6 GB VRAM).

Usage:
    python train_local.py                # full pipeline: parse → convert → train → evaluate
    python train_local.py --resume       # resume an interrupted training run
    python train_local.py --eval-only    # evaluate the best checkpoint without training
"""

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

from PIL import Image


# ──────────────────────────── Configuration ────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data" / "Oil Tanks"
YOLO_ROOT    = PROJECT_ROOT / "yolo_dataset_3class"
RUNS_DIR     = PROJECT_ROOT / "runs"
PATCH_DIR    = DATA_DIR / "image_patches"

TRAIN_SPLIT  = 0.85
RANDOM_SEED  = 42
IMG_SIZE     = 512
BATCH_SIZE   = 8        # safe for 6 GB VRAM with yolov8n @ 512
EPOCHS       = 100
PATIENCE     = 20       # early-stopping epochs
SAVE_PERIOD  = 5        # checkpoint every N epochs
MODEL_NAME   = "yolov8n.pt"
RUN_NAME     = "oil_tanks_3class_yolov8"

# Class mapping: label key in labels.json → YOLO class ID
CLASS_MAP = {
    "Floating Head Tank": 0,
    "Tank": 1,
    "Tank Cluster": 2,
}


# ──────────────────────────── Helpers ──────────────────────────────────

def corners_to_yolo(corners, img_w, img_h, class_id=0):
    """Convert 4-corner polygon [(x,y), …] to YOLO line."""
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    xc = ((x_min + x_max) / 2) / img_w
    yc = ((y_min + y_max) / 2) / img_h
    w  = (x_max - x_min) / img_w
    h  = (y_max - y_min) / img_h
    return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def bbox_to_yolo(bbox, img_w, img_h, class_id=0):
    """Convert [x_min, y_min, x_max, y_max] to YOLO line."""
    x_min, y_min, x_max, y_max = bbox
    xc = ((x_min + x_max) / 2) / img_w
    yc = ((y_min + y_max) / 2) / img_h
    w  = (x_max - x_min) / img_w
    h  = (y_max - y_min) / img_h
    return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


# ─────────────────── Stage 1: Parse labels.json ───────────────────────

def parse_labels():
    """Load labels.json, filter out 'Skip' images, return usable entries with any tank class."""
    labels_path = DATA_DIR / "labels.json"
    if not labels_path.exists():
        sys.exit(f"ERROR: {labels_path} not found. Download the dataset first.")

    with open(labels_path) as f:
        labels = json.load(f)

    print(f"Total labeled entries: {len(labels)}")

    # Usable = any entry whose label is a dict containing at least one known class
    usable = []
    class_counts = {name: 0 for name in CLASS_MAP}
    for e in labels:
        if not isinstance(e.get("label"), dict):
            continue
        has_known = False
        for class_name in CLASS_MAP:
            if class_name in e["label"]:
                has_known = True
                class_counts[class_name] += len(e["label"][class_name])
        if has_known:
            usable.append(e)

    skipped = [e for e in labels if e.get("label") == "Skip"]

    print(f"Usable images (contain tanks): {len(usable)}")
    print(f"Skipped images (no tanks): {len(skipped)}")
    print(f"\nAnnotation counts per class:")
    for name, count in class_counts.items():
        print(f"  {CLASS_MAP[name]}: {name} — {count} annotations")

    if not usable:
        print("\nWARNING: No usable entries found.")
        sys.exit("Please inspect labels.json structure.")

    return usable


# ──────────── Stage 2: Convert to YOLO format + split ─────────────────

def convert_and_split(usable):
    """Convert entries to YOLO .txt labels and split into train/val."""
    random.seed(RANDOM_SEED)
    random.shuffle(usable)

    split_idx = int(TRAIN_SPLIT * len(usable))
    train_entries = usable[:split_idx]
    val_entries   = usable[split_idx:]
    print(f"\nSplit → Train: {len(train_entries)} | Val: {len(val_entries)}")

    # Create directory structure
    for split in ["train", "val"]:
        (YOLO_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    def process_split(entries, split_name):
        written = 0
        for entry in entries:
            fname = entry.get("file_name")
            if fname is None:
                continue

            src_img = PATCH_DIR / fname
            if not src_img.exists():
                continue

            with Image.open(src_img) as im:
                img_w, img_h = im.size

            lines = []
            # Process all 3 classes from the label dict
            for class_name, class_id in CLASS_MAP.items():
                tank_list = entry["label"].get(class_name, [])
                for tank in tank_list:
                    geometry = tank.get("geometry", [])
                    if len(geometry) < 4:
                        continue
                    xs = [pt["x"] for pt in geometry]
                    ys = [pt["y"] for pt in geometry]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    xc = ((x_min + x_max) / 2) / img_w
                    yc = ((y_min + y_max) / 2) / img_h
                    w  = (x_max - x_min) / img_w
                    h  = (y_max - y_min) / img_h
                    lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            if not lines:
                continue

            shutil.copy2(src_img, YOLO_ROOT / "images" / split_name / fname)
            label_path = YOLO_ROOT / "labels" / split_name / (Path(fname).stem + ".txt")
            label_path.write_text("\n".join(lines))
            written += 1
        return written

    n_train = process_split(train_entries, "train")
    n_val   = process_split(val_entries, "val")
    print(f"Written → train: {n_train}, val: {n_val}")

    if n_train == 0:
        sys.exit("ERROR: No training images were written. Check PATCH_DIR and labels.json structure.")

    return n_train, n_val


# ──────────────── Stage 3: Create data.yaml ───────────────────────────

def create_data_yaml():
    """Write the YOLO data.yaml config file."""
    yaml_content = f"""path: {YOLO_ROOT}
train: images/train
val: images/val

names:
  0: floating_head_tank
  1: fixed_roof_tank
  2: tank_cluster
"""
    yaml_path = YOLO_ROOT / "data.yaml"
    yaml_path.write_text(yaml_content)
    print(f"\ndata.yaml written to {yaml_path}")
    print(yaml_content)
    return yaml_path


# ──────────────── Stage 4: Train ──────────────────────────────────────

def train(data_yaml, resume=False):
    """Train YOLOv8 or resume from last checkpoint."""
    from ultralytics import YOLO

    if resume:
        resume_path = RUNS_DIR / RUN_NAME / "weights" / "last.pt"
        if not resume_path.exists():
            sys.exit(f"ERROR: No checkpoint found at {resume_path}")
        print(f"\nResuming training from {resume_path}")
        model = YOLO(str(resume_path))
        model.train(resume=True)
    else:
        print(f"\nStarting training: {MODEL_NAME} | {EPOCHS} epochs | batch={BATCH_SIZE} | imgsz={IMG_SIZE}")
        model = YOLO(MODEL_NAME)
        model.train(
            data=str(data_yaml),
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            project=str(RUNS_DIR),
            name=RUN_NAME,
            patience=PATIENCE,
            save_period=SAVE_PERIOD,
            exist_ok=True,
            device=0,
        )

    return model


# ──────────────── Stage 5: Evaluate ───────────────────────────────────

def evaluate(model=None):
    """Run validation and print metrics."""
    from ultralytics import YOLO

    if model is None:
        best_path = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
        if not best_path.exists():
            sys.exit(f"ERROR: No best weights found at {best_path}")
        model = YOLO(str(best_path))

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    metrics = model.val()
    print(f"  mAP50:      {metrics.box.map50:.4f}")
    print(f"  mAP50-95:   {metrics.box.map:.4f}")
    print(f"  Precision:  {metrics.box.mp:.4f}")
    print(f"  Recall:     {metrics.box.mr:.4f}")
    print("=" * 60)
    return metrics


# ──────────────── Stage 6: Export best weights ────────────────────────

def export_weights():
    """Copy best.pt to project root for easy access."""
    src = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
    dst = PROJECT_ROOT / "best_oil_tanks_3class.pt"
    if src.exists():
        shutil.copy2(src, dst)
        print(f"\n✓ Best weights exported to: {dst}")
    else:
        print(f"\nWARNING: {src} not found — training may not have completed.")


# ──────────────── Main ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Oil Tank Detection — Local YOLOv8 Training")
    parser.add_argument("--resume",    action="store_true", help="Resume interrupted training")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate best checkpoint only")
    args = parser.parse_args()

    # ── Verify GPU ──
    import torch
    print(f"PyTorch {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("WARNING: CUDA not available — training will run on CPU (very slow).")
        resp = input("Continue on CPU? [y/N] ").strip().lower()
        if resp != "y":
            sys.exit("Aborted. Install PyTorch with CUDA support first.")

    if args.eval_only:
        evaluate()
        return

    if args.resume:
        model = train(None, resume=True)
        evaluate(model)
        export_weights()
        return

    # ── Full pipeline ──
    print("\n" + "=" * 60)
    print("STAGE 1: Parsing labels.json")
    print("=" * 60)
    usable = parse_labels()

    print("\n" + "=" * 60)
    print("STAGE 2: Converting to YOLO format + train/val split")
    print("=" * 60)
    convert_and_split(usable)

    print("\n" + "=" * 60)
    print("STAGE 3: Creating data.yaml")
    print("=" * 60)
    data_yaml = create_data_yaml()

    print("\n" + "=" * 60)
    print("STAGE 4: Training YOLOv8")
    print("=" * 60)
    model = train(data_yaml)

    print("\n" + "=" * 60)
    print("STAGE 5: Evaluation")
    print("=" * 60)
    evaluate(model)

    print("\n" + "=" * 60)
    print("STAGE 6: Exporting best weights")
    print("=" * 60)
    export_weights()

    print("\n✓ Pipeline complete!")


if __name__ == "__main__":
    main()
