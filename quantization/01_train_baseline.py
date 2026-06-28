"""
Phase 1: Train YOLOv8n baseline on People Detection dataset.
Run this first — all subsequent scripts depend on runs/baseline/weights/best.pt
"""

import os
from pathlib import Path
from ultralytics import YOLO

DATA_YAML   = Path(__file__).parent / "data.yaml"
RUNS_DIR    = Path(__file__).parent / "runs"
PROJECT_DIR = RUNS_DIR / "baseline"

def train():
    model = YOLO("yolov8n.pt")

    results = model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=16,
        project=str(RUNS_DIR),
        name="baseline",
        exist_ok=True,
        device=0,                # GPU 0; set to "cpu" if no CUDA
        workers=8,
        patience=20,             # early stop if no improvement for 20 epochs
        save=True,
        cache=True,
    )

    best = PROJECT_DIR / "weights" / "best.pt"
    print(f"\nTraining complete. Best model: {best}")
    return best


def validate(weights: Path):
    model = YOLO(str(weights))
    metrics = model.val(data=str(DATA_YAML), imgsz=640, batch=16, device=0)
    print(f"\nBaseline validation:")
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    best = train()
    validate(best)
