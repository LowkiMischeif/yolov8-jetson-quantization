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
# Proposed method plus baseline assignment variants (04b): every
# results/assignment_<name>.json found is built into an engine.
BUDGETS     = [
    "tight", "medium", "relaxed",
    "hawq_tight", "hawq_medium", "hawq_relaxed",
    "flopsproxy_tight", "flopsproxy_medium", "flopsproxy_relaxed",
]


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

def trt_conv_name_to_py_name(trt_name: str) -> str:
    """
    ONNX/TRT Conv node name → PyTorch named_modules() dotted path.

    Ultralytics ONNX export names every module level as its own path segment,
    and a child segment repeats its parent's scope within nn.Sequential /
    nn.ModuleList containers:
      '/model.0/conv/Conv'              → 'model.0.conv'
      '/model.22/cv2.0/cv2.0.2/Conv'    → 'model.22.cv2.0.2'
      '/model.22/cv2.2/cv2.2.0/conv/Conv' → 'model.22.cv2.2.0.conv'

    We therefore normalise by (1) stripping the trailing op segment ('Conv'),
    (2) collapsing any segment that extends the previous segment's dotted
    path, and (3) joining with dots.  This is exact-match safe (verified
    64/64 against Ultralytics 8.4 opset-17 export; run with --verify to
    re-check after any Ultralytics/opset upgrade).
    """
    name = trt_name.split(" + ")[0]           # defensive: drop fusion suffixes
    parts = [p for p in name.split("/") if p]
    if parts and parts[-1] == "Conv":
        parts = parts[:-1]
    merged: list = []
    for seg in parts:
        if merged and seg.startswith(merged[-1] + "."):
            merged[-1] = seg                  # child repeats parent scope
        else:
            merged.append(seg)
    return ".".join(merged)


def verify_matching_against_onnx(assignment_names: set) -> None:
    """
    Workstation-side check (no TensorRT needed): parse the ONNX graph and
    assert that every Conv node maps 1:1 onto the assignment's PyTorch names.
    Raises RuntimeError on any mismatch instead of silently falling back.
    """
    import onnx

    graph = onnx.load(str(ONNX_PATH)).graph
    onnx_convs = {trt_conv_name_to_py_name(n.name)
                  for n in graph.node if n.op_type == "Conv"}

    missing = sorted(assignment_names - onnx_convs)
    extra   = sorted(onnx_convs - assignment_names)
    if missing or extra:
        raise RuntimeError(
            "ONNX↔PyTorch Conv name mapping is NOT bijective.\n"
            f"  assignment names with no ONNX match ({len(missing)}): {missing[:5]} …\n"
            f"  ONNX convs with no assignment entry ({len(extra)}): {extra[:5]} …\n"
            "Refusing to build a mixed-precision engine with unmatched layers."
        )
    print(f"  Verified: {len(onnx_convs)}/{len(assignment_names)} Conv layers "
          "map bijectively between ONNX graph and assignment JSON.")


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

    n_fp16 = n_int8 = n_auto = 0
    matched: set = set()
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type != trt.LayerType.CONVOLUTION:
            n_auto += 1            # non-Conv operators: TRT automatic precision
            continue

        py_name   = trt_conv_name_to_py_name(layer.name)
        precision = assignment.get(py_name)
        if precision is None:
            raise RuntimeError(
                f"Conv layer '{layer.name}' (normalised '{py_name}') has no "
                "entry in the assignment JSON — mapping is broken, aborting "
                "instead of silently falling back to TRT auto precision."
            )
        matched.add(py_name)

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
            raise ValueError(f"Unknown precision '{precision}' for {py_name}")

    unmatched = set(assignment) - matched
    if unmatched:
        raise RuntimeError(
            f"{len(unmatched)} assignment layers were never seen in the TRT "
            f"network: {sorted(unmatched)[:5]} … — aborting."
        )
    print(f"  Layer assignments — FP16: {n_fp16}  INT8: {n_int8}  "
          f"TRT-auto (non-Conv): {n_auto}  "
          f"[matched {len(matched)}/{len(assignment)} Conv2d]")

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

    dump_engine_precisions(serialized, budget_name)
    return out_path


def dump_engine_precisions(serialized, budget_name: str) -> None:
    """
    Deserialize the freshly built engine and record the *realised* per-layer
    precisions via IEngineInspector — deployment-side proof that the requested
    mixed-precision constraints were honoured (or where TRT overrode them
    under PREFER_PRECISION_CONSTRAINTS).
    """
    import tensorrt as trt

    logger  = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine  = runtime.deserialize_cuda_engine(bytes(serialized))
    if engine is None:
        print("  WARNING: could not deserialize engine for inspection.")
        return

    inspector = engine.create_engine_inspector()
    info = json.loads(
        inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    )
    out_path = RESULTS_DIR / f"engine_inspect_{budget_name}.json"
    with open(out_path, "w") as f:
        json.dump(info, f, indent=2)

    counts: dict = {}
    for layer in info.get("Layers", []):
        prec = (layer.get("Precision")
                or layer.get("LayerType", "?")) if isinstance(layer, dict) else "?"
        counts[prec] = counts.get(prec, 0) + 1
    print(f"  Engine inspector precision histogram: {counts}")
    print(f"  Full per-layer report: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import sys
    verify_only = "--verify" in sys.argv

    if not WEIGHTS.exists() and not verify_only:
        raise FileNotFoundError(
            f"Weights not found at {WEIGHTS}. "
            "Transfer best.pt from workstation and run 05_export.py first."
        )

    ENGINES_DIR.mkdir(parents=True, exist_ok=True)
    if not verify_only:
        ensure_onnx()

    built = []
    for budget in BUDGETS:
        json_path = RESULTS_DIR / f"assignment_{budget}.json"
        if not json_path.exists():
            print(f"SKIP {budget}: {json_path} not found. Run 04_assignment.py first.")
            continue

        with open(json_path) as f:
            assignment = json.load(f)

        # Fail fast if the ONNX↔PyTorch name mapping is broken (workstation-
        # checkable: `python 05b_export_mixed.py --verify` needs no TensorRT).
        verify_matching_against_onnx(set(assignment.keys()))
        if verify_only:
            print(f"  [{budget}] mapping OK — verify-only mode, skipping build.")
            continue

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
