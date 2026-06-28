"""
Phase 3b: Per-layer latency measurement on the Jetson Orin Nano Super.

Run this ON THE JETSON (after transferring best.pt) instead of 03_latency_proxy.py.
Uses the same PyTorch CUDA-event timing method so layer names are identical to
layer_sensitivity.csv — this is required for 04_assignment.py to merge correctly.

trtexec --dumpProfile (06_jetson_profile.sh) outputs fused TRT layer names that
do NOT match PyTorch module names; use this script for the assignment input.

Output: results/layer_latency_jetson.csv
"""

import csv
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO

WEIGHTS     = Path(__file__).parent / "runs" / "baseline" / "weights" / "best.pt"
RESULTS_DIR = Path(__file__).parent / "results"
OUT_CSV     = RESULTS_DIR / "layer_latency_jetson.csv"

RESULTS_DIR.mkdir(exist_ok=True)

INPUT_SIZE = (1, 3, 640, 640)
WARMUP     = 20
REPEATS    = 100


def measure_layer_latency(module: nn.Conv2d, sample_input: torch.Tensor,
                           device: torch.device) -> float:
    module = module.to(device)
    x = sample_input.to(device)

    with torch.no_grad():
        for _ in range(WARMUP):
            _ = module(x)

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        start.record()
        for _ in range(REPEATS):
            _ = module(x)
        end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end) / REPEATS


def capture_input_shape(module: nn.Conv2d, full_model: nn.Module) -> tuple:
    captured = {}

    def hook(m, inp, out):
        captured["shape"] = inp[0].shape

    handle = module.register_forward_hook(hook)
    dummy = torch.zeros(INPUT_SIZE)
    try:
        with torch.no_grad():
            full_model(dummy)
    except Exception:
        pass
    finally:
        handle.remove()

    return captured.get("shape", (1, module.in_channels, 20, 20))


def main():
    if not WEIGHTS.exists():
        raise FileNotFoundError(
            f"Weights not found at {WEIGHTS}. "
            "Transfer best.pt from workstation first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model    = YOLO(str(WEIGHTS))
    pt_model = model.model.eval().to(device)

    conv_layers = [
        (name, mod)
        for name, mod in pt_model.named_modules()
        if isinstance(mod, nn.Conv2d)
    ]
    print(f"Found {len(conv_layers)} Conv2d layers.\n")

    rows = []
    for i, (name, mod) in enumerate(conv_layers):
        input_shape = capture_input_shape(mod, pt_model)
        sample      = torch.zeros(input_shape)
        latency_ms  = measure_layer_latency(mod, sample, device)

        rows.append({
            "layer_idx":    i,
            "layer_name":   name,
            "in_channels":  mod.in_channels,
            "out_channels": mod.out_channels,
            "kernel_size":  mod.kernel_size[0],
            "input_h":      input_shape[-2],
            "input_w":      input_shape[-1],
            "latency_ms":   round(latency_ms, 6),
            "source":       "jetson_pytorch",
        })
        print(f"[{i+1:3d}/{len(conv_layers)}] {name:<60s}  {latency_ms:.4f} ms")

    fieldnames = ["layer_idx", "layer_name", "in_channels", "out_channels",
                  "kernel_size", "input_h", "input_w", "latency_ms", "source"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = sum(r["latency_ms"] for r in rows)
    print(f"\nTotal Conv2d latency (sum): {total:.3f} ms")
    print(f"Results saved to {OUT_CSV}")
    print("\nRun 04_assignment.py next — it will pick up layer_latency_jetson.csv automatically.")


if __name__ == "__main__":
    main()
