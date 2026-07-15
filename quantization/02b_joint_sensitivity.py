"""
Phase 2b: Multi-layer JOINT sensitivity analysis (GPU workstation).

Addresses the cross-layer-coupling question: single-layer one-at-a-time
sensitivity (02_sensitivity_analysis.py) assumes per-layer accuracy drops
are additive.  Here we quantize *sets* of layers simultaneously and compare
the measured joint mAP drop against the sum of the single-layer deltas.

Configurations evaluated:
  1. The INT8-assigned layer set of each budget (tight / medium / relaxed)
     from results/assignment_<budget>.json.
  2. A cumulative top-k sweep (k = 1, 2, 4, 8, 16, 32, 64) over layers
     ranked by descending single-layer delta.

Output: results/joint_sensitivity.csv
    config, n_layers, sum_single_delta, joint_delta, additivity_error

Usage:
    python 02b_joint_sensitivity.py [--weights PATH] [--quick]
"""

import argparse
import csv
import json
from pathlib import Path

import torch.nn as nn
from ultralytics import YOLO

BASE        = Path(__file__).parent
DATA_YAML   = BASE / "data.yaml"
WEIGHTS     = BASE / "runs" / "baseline" / "weights" / "best.pt"
RESULTS_DIR = BASE / "results"
SENS_CSV    = RESULTS_DIR / "layer_sensitivity.csv"
OUT_CSV     = RESULTS_DIR / "joint_sensitivity.csv"

TOPK_SWEEP = [1, 2, 4, 8, 16, 32, 64]

# Reuse the identical INT8 simulation from Phase 2 so joint and single-layer
# deltas are directly comparable.
import importlib.util
_spec = importlib.util.spec_from_file_location("m02", BASE / "02_sensitivity_analysis.py")
_m02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m02)
fake_int8 = _m02.fake_int8


def run_val(model: YOLO) -> float:
    metrics = model.val(data=str(DATA_YAML), imgsz=640, batch=16,
                        device=0, verbose=False)
    return float(metrics.box.map)  # mAP50-95


def quantize_hook(module, inputs, output):
    return fake_int8(output.detach())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: only k=1 and the tight budget set")
    args = ap.parse_args()

    with open(SENS_CSV, newline="") as f:
        sens_rows = list(csv.DictReader(f))
    single_delta = {r["layer_name"]: float(r["delta_map"]) for r in sens_rows}
    ranked = sorted(single_delta, key=single_delta.get, reverse=True)

    model = YOLO(args.weights)
    conv_modules = {
        name: module
        for name, module in model.model.named_modules()
        if isinstance(module, nn.Conv2d)
    }
    missing = set(single_delta) - set(conv_modules)
    if missing:
        raise RuntimeError(f"Layers in sensitivity CSV not found in model: {missing}")

    print("Baseline validation …")
    baseline = run_val(model)
    print(f"Baseline mAP50-95: {baseline:.5f}\n")

    configs = []
    for budget in ["tight", "medium", "relaxed"]:
        path = RESULTS_DIR / f"assignment_{budget}.json"
        if path.exists():
            with open(path) as f:
                assignment = json.load(f)
            layer_set = [n for n, p in assignment.items() if p == "INT8"]
            configs.append((f"budget_{budget}", layer_set))
    for k in TOPK_SWEEP:
        configs.append((f"top{k}", ranked[:k]))

    if args.quick:
        configs = [c for c in configs if c[0] in ("budget_tight", "top1")]

    rows = []
    for name, layer_set in configs:
        hooks = [conv_modules[l].register_forward_hook(quantize_hook)
                 for l in layer_set]
        try:
            joint_map = run_val(model)
        finally:
            for h in hooks:
                h.remove()

        joint_delta = baseline - joint_map
        sum_single  = sum(single_delta[l] for l in layer_set)
        rows.append({
            "config":            name,
            "n_layers":          len(layer_set),
            "sum_single_delta":  round(sum_single, 6),
            "joint_delta":       round(joint_delta, 6),
            "additivity_error":  round(joint_delta - sum_single, 6),
        })
        print(f"[{name:>16s}] n={len(layer_set):2d}  "
              f"sum-single={sum_single:+.5f}  joint={joint_delta:+.5f}  "
              f"err={joint_delta - sum_single:+.5f}")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {OUT_CSV}")


if __name__ == "__main__":
    main()
