"""
Phase 2: Per-layer INT8 sensitivity analysis (runs on GPU workstation, no Jetson needed).

For each Conv2d layer in YOLOv8n:
  - Attach a forward hook that simulates INT8 quantization on that layer's output
  - Evaluate mAP50-95 on the validation set
  - Record delta_mAP = baseline_mAP - quantized_mAP

Output: results/layer_sensitivity.csv
"""

import csv
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO

DATA_YAML   = Path(__file__).parent / "data.yaml"
WEIGHTS     = Path(__file__).parent / "runs" / "baseline" / "weights" / "best.pt"
RESULTS_DIR = Path(__file__).parent / "results"
OUT_CSV     = RESULTS_DIR / "layer_sensitivity.csv"

RESULTS_DIR.mkdir(exist_ok=True)


# ── INT8 simulation ──────────────────────────────────────────────────────────

def fake_int8(tensor: torch.Tensor) -> torch.Tensor:
    """Simulate symmetric per-tensor INT8 max calibration."""
    if tensor.numel() == 0:
        return tensor
    scale = tensor.abs().max() / 127.0
    if scale == 0:
        return tensor
    return (tensor / scale).round().clamp(-128, 127) * scale


def make_hook(store: dict, name: str):
    """Forward hook that quantizes the layer output to INT8."""
    def hook(module, input, output):
        store["output"] = fake_int8(output.detach())
        return store["output"]
    return hook


# ── Validation helper ─────────────────────────────────────────────────────────

def run_val(model: YOLO) -> float:
    metrics = model.val(
        data=str(DATA_YAML),
        imgsz=640,
        batch=16,
        device=0,
        verbose=False,
    )
    return float(metrics.box.map)  # mAP50-95


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading model and computing baseline mAP...")
    model = YOLO(str(WEIGHTS))
    baseline_map = run_val(model)
    print(f"Baseline mAP50-95: {baseline_map:.4f}\n")

    # Collect all Conv2d layers with their full dotted names
    pytorch_model = model.model
    conv_layers = [
        (name, module)
        for name, module in pytorch_model.named_modules()
        if isinstance(module, nn.Conv2d)
    ]
    print(f"Found {len(conv_layers)} Conv2d layers to analyse.\n")

    rows = []
    store = {}

    for i, (name, module) in enumerate(conv_layers):
        hook = module.register_forward_hook(make_hook(store, name))
        try:
            quant_map = run_val(model)
        finally:
            hook.remove()

        delta = baseline_map - quant_map
        rows.append({
            "layer_idx":    i,
            "layer_name":   name,
            "in_channels":  module.in_channels,
            "out_channels": module.out_channels,
            "kernel_size":  module.kernel_size[0],
            "baseline_map": round(baseline_map, 6),
            "quant_map":    round(quant_map, 6),
            "delta_map":    round(delta, 6),
        })
        print(f"[{i+1:3d}/{len(conv_layers)}] {name:<60s}  delta={delta:+.4f}")

    # Write CSV
    fieldnames = ["layer_idx", "layer_name", "in_channels", "out_channels",
                  "kernel_size", "baseline_map", "quant_map", "delta_map"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSensitivity analysis complete. Results saved to {OUT_CSV}")
    most_sensitive = sorted(rows, key=lambda r: r["delta_map"], reverse=True)[:5]
    print("\nTop 5 most sensitive layers:")
    for r in most_sensitive:
        print(f"  {r['layer_name']:<60s}  delta={r['delta_map']:+.4f}")


if __name__ == "__main__":
    main()
