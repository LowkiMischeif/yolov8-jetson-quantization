"""
Phase 5: Export best.pt to ONNX and TensorRT engines (FP32, FP16, INT8 PTQ).

Run on the Jetson Orin Nano Super (TensorRT must be installed via JetPack).
Transfer best.pt from your workstation first:
    scp runs/baseline/weights/best.pt user@jetson:/path/to/quantization/runs/baseline/weights/

Also samples the INT8 calibration holdout (500 images) from the training set.

Outputs (in runs/engines/):
  yolov8n_fp32.engine
  yolov8n_fp16.engine
  yolov8n_int8_ptq.engine
  yolov8n_int8_ptq.cache    (calibration table)
  calibration_images/       (500-image holdout)
"""

import random
import shutil
from pathlib import Path

from ultralytics import YOLO

WEIGHTS        = Path(__file__).parent / "runs" / "baseline" / "weights" / "best.pt"
ENGINES_DIR    = Path(__file__).parent / "runs" / "engines"
CALIB_DIR      = Path(__file__).parent / "runs" / "calibration_images"
TRAIN_IMAGES   = Path(__file__).parent / "runs" / "calibration_images"  # pre-populated from workstation
N_CALIB        = 500
IMGSZ          = 640

ENGINES_DIR.mkdir(parents=True, exist_ok=True)


def sample_calibration_images():
    """Use pre-transferred calibration images from runs/calibration_images/."""
    all_images = list(TRAIN_IMAGES.glob("*.jpg")) + list(TRAIN_IMAGES.glob("*.png"))
    if len(all_images) == 0:
        raise RuntimeError(
            f"No calibration images found in {TRAIN_IMAGES}.\n"
            "Transfer images from workstation:\n"
            "  scp workstation:C:/Papers/Vision/.../train/images/*.jpg ~/quantization/runs/calibration_images/"
        )
    # Copy to CALIB_DIR if different location
    if TRAIN_IMAGES != CALIB_DIR:
        CALIB_DIR.mkdir(parents=True, exist_ok=True)
        for img in all_images[:N_CALIB]:
            shutil.copy(img, CALIB_DIR / img.name)
    print(f"Using {min(len(all_images), N_CALIB)} calibration images from {TRAIN_IMAGES}")


def export_fp32():
    print("\n--- Exporting FP32 ONNX + TensorRT engine ---")
    model = YOLO(str(WEIGHTS))
    model.export(
        format="engine",
        imgsz=IMGSZ,
        half=False,
        int8=False,
        device=0,
        workspace=4,
        simplify=True,
    )
    src = WEIGHTS.parent / "best.engine"
    dst = ENGINES_DIR / "yolov8n_fp32.engine"
    if src.exists():
        shutil.move(str(src), str(dst))
    print(f"FP32 engine saved to {dst}")


def export_fp16():
    print("\n--- Exporting FP16 TensorRT engine ---")
    model = YOLO(str(WEIGHTS))
    model.export(
        format="engine",
        imgsz=IMGSZ,
        half=True,
        int8=False,
        device=0,
        workspace=4,
        simplify=True,
    )
    src = WEIGHTS.parent / "best.engine"
    dst = ENGINES_DIR / "yolov8n_fp16.engine"
    if src.exists():
        shutil.move(str(src), str(dst))
    print(f"FP16 engine saved to {dst}")


def export_int8_ptq():
    print("\n--- Exporting INT8 PTQ TensorRT engine ---")
    sample_calibration_images()
    model = YOLO(str(WEIGHTS))
    model.export(
        format="engine",
        imgsz=IMGSZ,
        half=False,
        int8=True,
        data=str(Path(__file__).parent / "data.yaml"),
        device=0,
        workspace=4,
        simplify=True,
    )
    src = WEIGHTS.parent / "best.engine"
    dst = ENGINES_DIR / "yolov8n_int8_ptq.engine"
    if src.exists():
        shutil.move(str(src), str(dst))
    print(f"INT8 PTQ engine saved to {dst}")


def export_reference_model(model_name: str, dst_name: str):
    """Export a pretrained YOLOv8 model to FP32 TensorRT engine.

    Used for YOLOv8s and YOLOv8x latency/accuracy reference conditions.
    Ultralytics downloads the weights on first call and saves the engine
    alongside the cached .pt file; we move it to runs/engines/.
    """
    dst = ENGINES_DIR / f"{dst_name}.engine"
    if dst.exists():
        print(f"\n{model_name}: already exported at {dst}, skipping.")
        return

    print(f"\n--- Exporting {model_name} FP32 TensorRT engine ---")
    model    = YOLO(model_name)
    exported = model.export(
        format="engine",
        imgsz=IMGSZ,
        half=False,
        int8=False,
        device=0,
        workspace=4,
        simplify=True,
    )
    src = Path(str(exported)) if exported else None

    # Fallback: engine saved next to the cached .pt (same stem)
    if src is None or not src.exists():
        src = Path(model.ckpt_path).with_suffix(".engine")

    if src.exists():
        shutil.move(str(src), str(dst))
        print(f"{model_name} FP32 engine saved to {dst}")
    else:
        print(f"WARNING: Could not locate exported engine for {model_name} "
              f"(tried {src}).")


if __name__ == "__main__":
    if not WEIGHTS.exists():
        raise FileNotFoundError(
            f"Weights not found at {WEIGHTS}. "
            "Run 01_train_baseline.py first, then transfer best.pt to Jetson."
        )

    export_fp32()
    export_fp16()
    export_int8_ptq()

    print("\n--- Exporting reference models (YOLOv8s, YOLOv8x) ---")
    export_reference_model("yolov8s.pt", "yolov8s_fp32")
    export_reference_model("yolov8x.pt", "yolov8x_fp32")

    print("\nAll engines exported:")
    for e in sorted(ENGINES_DIR.glob("*.engine")):
        size_mb = e.stat().st_size / 1024 / 1024
        print(f"  {e.name:<35s}  {size_mb:.1f} MB")
