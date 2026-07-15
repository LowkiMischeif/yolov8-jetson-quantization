"""
Phase 7: Structured benchmark of all engine conditions on the Jetson.
Run ON THE JETSON after all engines are built.

Measures, per engine:
  - mAP50 / mAP50-95 by validating THE COMPILED TRT ENGINE ITSELF
    (Ultralytics loads .engine files directly) on the full validation split
  - latency P50 / P95 and throughput from trtexec, repeated N_RUNS times,
    reported as median +/- std across runs
  - GPU memory and serialized engine size

Also records the device power/clock configuration (nvpmodel, jetson_clocks)
to results/jetson_env.txt so the benchmark is reproducible.

Output: results/results_main.csv  (one row per condition, with variance)
"""

import csv
import statistics
import subprocess
import re
from pathlib import Path

from ultralytics import YOLO

BASE        = Path(__file__).parent
DATA_YAML   = BASE / "data_jetson.yaml"   # must point at the FULL val split
ENGINES_DIR = BASE / "runs" / "engines"
RESULTS_DIR = BASE / "results"
OUT_CSV     = RESULTS_DIR / "results_main.csv"
ENV_TXT     = RESULTS_DIR / "jetson_env.txt"
TRTEXEC     = "/usr/src/tensorrt/bin/trtexec"

RESULTS_DIR.mkdir(exist_ok=True)

IMGSZ   = 640
WARMUP  = 100
ITERS   = 200
N_RUNS  = 5      # repeated trtexec runs for run-to-run variance

# (label, engine_filename)
CONDITIONS = [
    ("YOLOv8n FP32 (baseline)",        "yolov8n_fp32.engine"),
    ("YOLOv8n FP16",                   "yolov8n_fp16.engine"),
    ("YOLOv8n INT8 PTQ (uniform)",     "yolov8n_int8_ptq.engine"),
    ("YOLOv8n Mixed-PTQ tight",        "yolov8n_mixed_tight.engine"),
    ("YOLOv8n Mixed-PTQ medium",       "yolov8n_mixed_medium.engine"),
    ("YOLOv8n Mixed-PTQ relaxed",      "yolov8n_mixed_relaxed.engine"),
    # HAWQ-style Hessian-trace baseline assignments (04b)
    ("YOLOv8n Mixed-HAWQ tight",       "yolov8n_mixed_hawq_tight.engine"),
    ("YOLOv8n Mixed-HAWQ medium",      "yolov8n_mixed_hawq_medium.engine"),
    ("YOLOv8n Mixed-HAWQ relaxed",     "yolov8n_mixed_hawq_relaxed.engine"),
    # FLOPs-proxy latency ablation assignments (04b)
    ("YOLOv8n Mixed-FLOPsProxy tight",   "yolov8n_mixed_flopsproxy_tight.engine"),
    ("YOLOv8n Mixed-FLOPsProxy medium",  "yolov8n_mixed_flopsproxy_medium.engine"),
    ("YOLOv8n Mixed-FLOPsProxy relaxed", "yolov8n_mixed_flopsproxy_relaxed.engine"),
]


def record_environment() -> None:
    """Save power mode and clock state alongside the results."""
    blocks = []
    for cmd in (["nvpmodel", "-q"],
                ["jetson_clocks", "--show"],
                ["cat", "/etc/nv_tegra_release"],
                ["uname", "-a"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=30)
            blocks.append(f"$ {' '.join(cmd)}\n{out.stdout}{out.stderr}")
        except Exception as exc:
            blocks.append(f"$ {' '.join(cmd)}\n(unavailable: {exc})")
    ENV_TXT.write_text("\n".join(blocks))
    print(f"Device environment recorded to {ENV_TXT}")


def run_trtexec_once(engine_path: Path) -> dict:
    cmd = [
        TRTEXEC,
        f"--loadEngine={engine_path}",
        f"--warmUp={WARMUP}",
        f"--iterations={ITERS}",
        "--percentile=50,95",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr

    def find(pattern):
        m = re.search(pattern, output)
        return float(m.group(1)) if m else None

    return {
        "p50": find(r"percentile\(50%\)\s*=\s*([\d.]+)\s*ms"),
        "p95": find(r"percentile\(95%\)\s*=\s*([\d.]+)\s*ms"),
        "fps": find(r"Throughput:\s*([\d.]+)\s*qps"),
        "mem": find(r"(?:GPU Compute Memory|GPU Memory).*?:\s*([\d.]+)\s*MiB"),
    }


def run_trtexec_repeated(engine_path: Path) -> dict:
    """N_RUNS independent trtexec runs → median and std across runs."""
    runs = []
    for i in range(N_RUNS):
        r = run_trtexec_once(engine_path)
        if r["p50"] is not None:
            runs.append(r)
            print(f"    run {i + 1}/{N_RUNS}: "
                  f"P50={r['p50']:.3f} ms  fps={r['fps']:.1f}")
        else:
            print(f"    run {i + 1}/{N_RUNS}: parse failure")
    if not runs:
        return {k: None for k in
                ("latency_p50_ms", "latency_p50_std_ms", "latency_p95_ms",
                 "throughput_fps", "throughput_std_fps", "gpu_memory_mb")}

    p50s = [r["p50"] for r in runs]
    fps  = [r["fps"] for r in runs if r["fps"] is not None]
    return {
        "latency_p50_ms":     round(statistics.median(p50s), 4),
        "latency_p50_std_ms": round(statistics.stdev(p50s), 4) if len(p50s) > 1 else 0.0,
        "latency_p95_ms":     round(statistics.median(
                                  [r["p95"] for r in runs if r["p95"]]), 4),
        "throughput_fps":     round(statistics.median(fps), 2) if fps else None,
        "throughput_std_fps": round(statistics.stdev(fps), 2) if len(fps) > 1 else 0.0,
        "gpu_memory_mb":      next((r["mem"] for r in runs if r["mem"]), None),
    }


def run_map_on_engine(engine_path: Path) -> dict:
    """
    Validate the compiled TRT engine itself (NOT the .pt weights) on the
    full validation split.  This is the deployment-accuracy number.
    """
    try:
        model = YOLO(str(engine_path), task="detect")
        metrics = model.val(data=str(DATA_YAML), imgsz=IMGSZ, batch=1,
                            device=0, verbose=False)
        return {
            "map50":    round(float(metrics.box.map50), 4),
            "map50_95": round(float(metrics.box.map), 4),
        }
    except Exception as exc:
        print(f"  WARNING: engine mAP evaluation failed ({exc}).")
        return {"map50": None, "map50_95": None}


def main():
    record_environment()
    rows = []

    for label, engine_file in CONDITIONS:
        engine_path = ENGINES_DIR / engine_file
        if not engine_path.exists():
            print(f"SKIP: {label} — engine not found ({engine_path})")
            continue

        print(f"\n{'=' * 60}\nBenchmarking: {label}")

        print(f"  trtexec x {N_RUNS} …")
        trt_metrics = run_trtexec_repeated(engine_path)

        print("  Engine mAP on full validation split …")
        map_metrics = run_map_on_engine(engine_path)

        row = {
            "condition":   label,
            "engine_file": engine_file,
            **map_metrics,
            **trt_metrics,
            "model_size_mb": round(engine_path.stat().st_size / 1024 / 1024, 2),
        }
        rows.append(row)
        print(f"  mAP50={row['map50']}  mAP50-95={row['map50_95']}  "
              f"P50={row['latency_p50_ms']}±{row['latency_p50_std_ms']}ms  "
              f"FPS={row['throughput_fps']}±{row['throughput_std_fps']}")

    fieldnames = ["condition", "engine_file", "map50", "map50_95",
                  "latency_p50_ms", "latency_p50_std_ms", "latency_p95_ms",
                  "throughput_fps", "throughput_std_fps",
                  "gpu_memory_mb", "model_size_mb"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nAll results saved to {OUT_CSV}")


if __name__ == "__main__":
    main()
