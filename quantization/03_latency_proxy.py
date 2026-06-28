"""
Phase 3a: Compute a FLOPs-based latency proxy for each Conv2d layer.

Run this TODAY on your GPU workstation while waiting for the Jetson.
Tomorrow, run 03b_latency_jetson.py ON THE JETSON to measure real per-layer
latency with matching PyTorch layer names → layer_latency_jetson.csv.

04_assignment.py automatically prefers layer_latency_jetson.csv over this proxy.

Output: results/layer_latency_proxy.csv
"""

import csv
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO

WEIGHTS     = Path(__file__).parent / "runs" / "baseline" / "weights" / "best.pt"
RESULTS_DIR = Path(__file__).parent / "results"
OUT_CSV     = RESULTS_DIR / "layer_latency_proxy.csv"

RESULTS_DIR.mkdir(exist_ok=True)

INPUT_SIZE = (1, 3, 640, 640)   # standard YOLOv8 input
WARMUP     = 50
REPEATS    = 200


def measure_layer_latency(module: nn.Conv2d, sample_input: torch.Tensor,
                           device: torch.device) -> float:
    """Time a single Conv2d layer using CUDA events."""
    module = module.to(device)
    x = sample_input.to(device)

    # Warmup
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
    return start.elapsed_time(end) / REPEATS   # ms per forward pass


def estimate_input_shape(module: nn.Conv2d, full_model: nn.Module,
                         layer_name: str) -> tuple:
    """
    Estimate plausible input spatial size by hooking the real forward pass.
    Falls back to (1, in_channels, 20, 20) if hook fails.
    """
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

    if "shape" in captured:
        return captured["shape"]
    return (1, module.in_channels, 20, 20)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running latency proxy on: {device}\n")

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
        input_shape = estimate_input_shape(mod, pt_model, name)
        sample      = torch.zeros(input_shape)

        if device.type == "cuda":
            latency_ms = measure_layer_latency(mod, sample, device)
        else:
            # CPU timing fallback (less accurate but usable as proxy)
            import time
            mod_cpu = mod.cpu()
            x_cpu   = sample.cpu()
            with torch.no_grad():
                for _ in range(WARMUP):
                    mod_cpu(x_cpu)
                t0 = time.perf_counter()
                for _ in range(REPEATS):
                    mod_cpu(x_cpu)
                latency_ms = (time.perf_counter() - t0) / REPEATS * 1000

        rows.append({
            "layer_idx":     i,
            "layer_name":    name,
            "in_channels":   mod.in_channels,
            "out_channels":  mod.out_channels,
            "kernel_size":   mod.kernel_size[0],
            "input_h":       input_shape[-2],
            "input_w":       input_shape[-1],
            "latency_ms":    round(latency_ms, 6),
            "source":        "gpu_proxy",
        })
        print(f"[{i+1:3d}/{len(conv_layers)}] {name:<60s}  {latency_ms:.4f} ms")

    fieldnames = ["layer_idx", "layer_name", "in_channels", "out_channels",
                  "kernel_size", "input_h", "input_w", "latency_ms", "source"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total = sum(r["latency_ms"] for r in rows)
    print(f"\nProxy total latency: {total:.2f} ms")
    print(f"Results saved to {OUT_CSV}")
    print("\nNOTE: Run 03b_latency_jetson.py on the Jetson to get real measured latency.")


if __name__ == "__main__":
    main()
