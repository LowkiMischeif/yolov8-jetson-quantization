"""
Phase 7: Structured benchmark of all engine conditions on the Jetson.
Run ON THE JETSON after all engines are built.

Measures: mAP50, mAP50-95, latency P50, latency P95, throughput, GPU memory, model size.

Conditions benchmarked:
  1. FP32 baseline
  2. FP16 TensorRT
  3. INT8 PTQ (uniform)
  4. Mixed-precision PTQ tight   (if assignment_tight engine exists)
  5. Mixed-precision PTQ medium  (if assignment_medium engine exists)
  6. Mixed-precision PTQ relaxed (if assignment_relaxed engine exists)
  Also: YOLOv8s FP32 and YOLOv8x FP32 as reference models.

Output: results/results_main.csv
"""

import csv
import subprocess
import re
from pathlib import Path

from ultralytics import YOLO

DATA_YAML   = Path(__file__).parent / "data_jetson.yaml"  # Jetson-local dataset
ENGINES_DIR = Path(__file__).parent / "runs" / "engines"
RESULTS_DIR = Path(__file__).parent / "results"
OUT_CSV     = RESULTS_DIR / "results_main.csv"
TRTEXEC     = "/usr/src/tensorrt/bin/trtexec"

RESULTS_DIR.mkdir(exist_ok=True)

IMGSZ    = 640
WARMUP   = 100   # reduced to avoid OOM on Jetson unified memory
ITERS    = 200

# Engines to benchmark: (label, engine_filename, weights_for_map)
CONDITIONS = [
    ("YOLOv8x FP32 (teacher)",      "yolov8x_fp32.engine",         "yolov8x.pt"),
    ("YOLOv8s FP32 (reference)",     "yolov8s_fp32.engine",         "yolov8s.pt"),
    ("YOLOv8n FP32 (baseline)",      "yolov8n_fp32.engine",         "runs/baseline/weights/best.pt"),
    ("YOLOv8n FP16",                 "yolov8n_fp16.engine",         "runs/baseline/weights/best.pt"),
    ("YOLOv8n INT8 PTQ (uniform)",   "yolov8n_int8_ptq.engine",     "runs/baseline/weights/best.pt"),
    ("YOLOv8n Mixed-PTQ tight",      "yolov8n_mixed_tight.engine",  "runs/baseline/weights/best.pt"),
    ("YOLOv8n Mixed-PTQ medium",     "yolov8n_mixed_medium.engine", "runs/baseline/weights/best.pt"),
    ("YOLOv8n Mixed-PTQ relaxed",    "yolov8n_mixed_relaxed.engine","runs/baseline/weights/best.pt"),
]


def run_trtexec(engine_path: Path) -> dict:
    """Run trtexec and parse latency / throughput metrics."""
    cmd = [
        TRTEXEC,
        f"--loadEngine={engine_path}",
        f"--warmUp={WARMUP}",
        f"--iterations={ITERS}",
        "--percentile=50,95",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    def find(pattern, default="N/A"):
        m = re.search(pattern, output)
        return m.group(1) if m else default

    # TRT 10.x format: percentile(50%) = X.XXX ms
    latency_p50  = find(r"percentile\(50%\)\s*=\s*([\d.]+)\s*ms")
    latency_p95  = find(r"percentile\(95%\)\s*=\s*([\d.]+)\s*ms")
    throughput   = find(r"Throughput:\s*([\d.]+)\s*qps")
    gpu_memory   = find(r"(?:GPU Compute Memory|GPU Memory).*?:\s*([\d.]+)\s*MiB")

    return {
        "latency_p50_ms": float(latency_p50) if latency_p50 != "N/A" else None,
        "latency_p95_ms": float(latency_p95) if latency_p95 != "N/A" else None,
        "throughput_fps": float(throughput)   if throughput  != "N/A" else None,
        "gpu_memory_mb":  float(gpu_memory)   if gpu_memory  != "N/A" else None,
    }


def run_map(weights_path) -> dict:
    """Evaluate mAP using Ultralytics val().

    Accepts:
      - None              → skip (returns None for both metrics)
      - relative path     → resolved against script directory
      - absolute path     → used as-is
      - bare model name   → e.g. "yolov8x.pt"; let Ultralytics find/download it
        (pretrained COCO weights: mAP is person-class only; label accordingly)
    """
    if weights_path is None:
        return {"map50": None, "map50_95": None}

    base_dir  = Path(__file__).parent
    is_bare   = "/" not in weights_path and "\\" not in weights_path
    full_path = (
        base_dir / weights_path
        if not Path(weights_path).is_absolute() and not is_bare
        else Path(weights_path)
    )

    if is_bare:
        print(f"  Loading pretrained model: {weights_path} (mAP = person class only)")
        model = YOLO(weights_path)
    elif full_path.exists():
        model = YOLO(str(full_path))
    else:
        print(f"  WARNING: weights not found at {full_path}, skipping mAP.")
        return {"map50": None, "map50_95": None}

    try:
        metrics = model.val(data=str(DATA_YAML), imgsz=IMGSZ, batch=1,
                            device=0, verbose=False)
        return {
            "map50":    round(float(metrics.box.map50), 4),
            "map50_95": round(float(metrics.box.map), 4),
        }
    except Exception as exc:
        print(f"  WARNING: mAP evaluation failed ({exc}). Skipping.")
        return {"map50": None, "map50_95": None}


def model_size_mb(engine_path: Path) -> float:
    return round(engine_path.stat().st_size / 1024 / 1024, 2) if engine_path.exists() else None


def main():
    rows = []

    for label, engine_file, weights in CONDITIONS:
        engine_path = ENGINES_DIR / engine_file
        if not engine_path.exists():
            print(f"SKIP: {label} — engine not found ({engine_path})")
            continue

        print(f"\n{'='*60}")
        print(f"Benchmarking: {label}")

        print("  Running trtexec...")
        trt_metrics = run_trtexec(engine_path)

        print("  Evaluating mAP...")
        map_metrics = run_map(weights)

        size = model_size_mb(engine_path)

        row = {
            "condition":      label,
            "engine_file":    engine_file,
            "map50":          map_metrics["map50"],
            "map50_95":       map_metrics["map50_95"],
            "latency_p50_ms": trt_metrics["latency_p50_ms"],
            "latency_p95_ms": trt_metrics["latency_p95_ms"],
            "throughput_fps": trt_metrics["throughput_fps"],
            "gpu_memory_mb":  trt_metrics["gpu_memory_mb"],
            "model_size_mb":  size,
        }
        rows.append(row)

        print(f"  mAP50={row['map50']}  mAP50-95={row['map50_95']}  "
              f"P50={row['latency_p50_ms']}ms  FPS={row['throughput_fps']}  "
              f"Mem={row['gpu_memory_mb']}MB  Size={row['model_size_mb']}MB")

    fieldnames = ["condition", "engine_file", "map50", "map50_95",
                  "latency_p50_ms", "latency_p95_ms", "throughput_fps",
                  "gpu_memory_mb", "model_size_mb"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nAll results saved to {OUT_CSV}")


if __name__ == "__main__":
    main()
