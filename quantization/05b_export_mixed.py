"""
Phase 5b: Build mixed-precision TensorRT engines from assignment JSONs.

Run ON THE JETSON after:
  - 05_export.py   (builds FP32/FP16/INT8 engines, calibration images)
  - 04_assignment.py (produces assignment_*.json with Jetson latency)

For each budget (tight, medium, relaxed):
  - Reads assignment_<budget>.json  (PyTorch layer_name → "FP16" | "INT8")
  - Builds a TensorRT engine with per-Conv2d precision constraints
  - INT8 layers: quantized compute, FP16 output activations
  - Non-Conv layers: TRT chooses precision automatically

Outputs:
  runs/engines/yolov8n_mixed_tight.engine
  runs/engines/yolov8n_mixed_medium.engine
  runs/engines/yolov8n_mixed_relaxed.engine

Requirements: tensorrt, torch, opencv-python, numpy
  All available on Jetson via JetPack + pip. No pycuda needed.
"""

import json
import shutil
import numpy as np
from pathlib import Path

from ultralytics import YOLO

WEIGHTS     = Path(__file__).parent / "runs" / "baseline" / "weights" / "best.pt"
ONNX_PATH   = Path(__file__).parent / "runs" / "baseline" / "weights" / "best.onnx"
ENGINES_DIR = Path(__file__).parent / "runs" / "engines"
CALIB_DIR   = Path(__file__).parent / "runs" / "calibration_images"
RESULTS_DIR = Path(__file__).parent / "results"
IMGSZ       = 640
BUDGETS     = ["tight", "medium", "relaxed"]


# ── ONNX export ───────────────────────────────────────────────────────────────

def ensure_onnx():
    if ONNX_PATH.exists():
        print(f"ONNX already present: {ONNX_PATH}")
        return
    print("Exporting best.pt → ONNX …")
    model = YOLO(str(WEIGHTS))
    model.export(format="onnx", imgsz=IMGSZ, simplify=True, opset=17)
    if not ONNX_PATH.exists():
        raise RuntimeError(f"ONNX export did not produce {ONNX_PATH}")
    print(f"ONNX saved: {ONNX_PATH}")


# ── Layer name mapping ────────────────────────────────────────────────────────

def py_name_to_trt_prefix(pytorch_name: str) -> str:
    """
    PyTorch named_modules() path → ONNX/TRT node name prefix.

    ONNX keeps dots before numeric indices but replaces dots before
    alphabetic segments with slashes:
      'model.0.conv'     →  '/model.0/conv'
      'model.22.cv2.0.2' →  '/model.22/cv2.0.2'

    TRT layer names append the op type ('/Conv') so we match on substring.
    """
    import re
    result = re.sub(r'\.(?=[a-zA-Z])', '/', pytorch_name)
    return '/' + result


# ── INT8 calibrator factory (TRT 10 requires explicit inheritance) ────────────

def make_calibrator(image_paths: list, imgsz: int, cache_name: str, engines_dir: Path):
    """Return an IInt8EntropyCalibrator2 instance using PyTorch CUDA tensors."""
    import tensorrt as trt
    import cv2
    import torch

    cache_path = engines_dir / f"calib_{cache_name}.cache"
    buf = torch.zeros((1, 3, imgsz, imgsz), dtype=torch.float32, device="cuda")
    paths = list(image_paths)

    class _Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self._idx = 0

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            while self._idx < len(paths):
                img = cv2.imread(str(paths[self._idx]))
                self._idx += 1
                if img is None:
                    continue
                img = cv2.resize(img, (imgsz, imgsz))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                tensor = torch.from_numpy(
                    np.ascontiguousarray(np.transpose(img, (2, 0, 1)))[np.newaxis]
                ).to("cuda")
                buf.copy_(tensor)
                return [buf.data_ptr()]
            return None

        def read_calibration_cache(self):
            if cache_path.exists():
                print(f"  Reusing calibration cache: {cache_path}")
                return cache_path.read_bytes()
            return None

        def write_calibration_cache(self, cache):
            cache_path.write_bytes(bytes(cache))
            print(f"  Calibration cache written: {cache_path}")

    return _Calibrator()


# ── Engine builder ────────────────────────────────────────────────────────────

def build_mixed_engine(assignment: dict, budget_name: str) -> Path:
    import tensorrt as trt

    out_path = ENGINES_DIR / f"yolov8n_mixed_{budget_name}.engine"

    logger  = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(ONNX_PATH, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX parse error [{i}]: {parser.get_error(i).desc()}")
            raise RuntimeError("ONNX parsing failed.")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1 << 30))  # 4 GiB

    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.INT8)
    # PREFER: TRT may override individual layers if the target precision is unsupported.
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

    # Build prefix → precision lookup from assignment JSON
    prefix_to_prec = {
        py_name_to_trt_prefix(k): v for k, v in assignment.items()
    }

    # Debug: show first 8 TRT layer names to verify prefix matching
    print("  Sample TRT layer names:")
    for i in range(min(8, network.num_layers)):
        print(f"    [{i}] {network.get_layer(i).name}")

    n_fp16 = n_int8 = n_unmatched = 0
    for i in range(network.num_layers):
        layer    = network.get_layer(i)
        trt_name = layer.name
        precision = None

        for prefix, prec in prefix_to_prec.items():
            if prefix in trt_name:
                precision = prec
                break

        if precision == "FP16":
            layer.precision = trt.DataType.HALF
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.DataType.HALF)
            n_fp16 += 1
        elif precision == "INT8":
            layer.precision = trt.DataType.INT8
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.DataType.HALF)
            n_int8 += 1
        else:
            n_unmatched += 1

    print(f"  Layer assignments — FP16: {n_fp16}  INT8: {n_int8}  "
          f"TRT-auto (non-Conv): {n_unmatched}")
    if n_fp16 == 0 and n_int8 == 0:
        print("  WARNING: No layers matched. Check TRT layer names above vs assignment keys.")

    # Calibrator required for INT8 layers
    calib_imgs = sorted(CALIB_DIR.glob("*.jpg")) + sorted(CALIB_DIR.glob("*.png"))
    if not calib_imgs:
        raise FileNotFoundError(
            f"No calibration images found in {CALIB_DIR}. "
            "Run 05_export.py first to sample them."
        )
    config.int8_calibrator = make_calibrator(calib_imgs, IMGSZ, budget_name, ENGINES_DIR)

    print("  Building engine (this may take several minutes) …")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"Engine build failed for budget '{budget_name}'.")

    with open(out_path, "wb") as f:
        f.write(bytes(serialized))

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"  Saved: {out_path}  ({size_mb:.1f} MB)")
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not WEIGHTS.exists():
        raise FileNotFoundError(
            f"Weights not found at {WEIGHTS}. "
            "Transfer best.pt from workstation and run 05_export.py first."
        )

    ENGINES_DIR.mkdir(parents=True, exist_ok=True)
    ensure_onnx()

    built = []
    for budget in BUDGETS:
        json_path = RESULTS_DIR / f"assignment_{budget}.json"
        if not json_path.exists():
            print(f"SKIP {budget}: {json_path} not found. Run 04_assignment.py first.")
            continue

        with open(json_path) as f:
            assignment = json.load(f)

        fp16_count = sum(1 for v in assignment.values() if v == "FP16")
        int8_count = sum(1 for v in assignment.values() if v == "INT8")

        print(f"\n{'='*60}")
        print(f"Budget: {budget}  (FP16={fp16_count}, INT8={int8_count})")
        built.append(build_mixed_engine(assignment, budget))

    if built:
        print("\nMixed-precision engines:")
        for p in built:
            print(f"  {p.name}  ({p.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print("\nNo engines built — check that assignment_*.json files exist.")


if __name__ == "__main__":
    main()
