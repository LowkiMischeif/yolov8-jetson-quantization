"""
Phase 4b: Baseline mixed-precision assignment strategies for comparison.

Two alternative assignment policies, evaluated under the SAME latency budgets
and built/benchmarked with the same 05b/07 pipeline as the proposed method:

  1. HAWQ-style Hessian-trace sensitivity (proxy, no on-device accuracy
     measurement): per-layer Hessian trace of the detection loss, estimated
     with Hutchinson probes (HAWQ-V2 average-trace criterion), replaces the
     measured one-at-a-time INT8 delta as the protection signal.
        -> results/hessian_traces.csv
        -> results/assignment_hawq_<budget>.json

  2. FLOPs-proxy latency ablation: the proposed measured-delta sensitivity,
     but with the offline FLOPs latency proxy (03_latency_proxy.py) in place
     of on-device measured latency.  Isolates the contribution of on-device
     profiling.
        -> results/assignment_flopsproxy_<budget>.json

Runs on the GPU workstation.  Usage:
    python 04b_assignment_baselines.py [--weights PATH] [--probes N] [--batches N]
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

BASE        = Path(__file__).parent
DATA_YAML   = BASE / "data.yaml"
WEIGHTS     = BASE / "runs" / "baseline" / "weights" / "best.pt"
RESULTS_DIR = BASE / "results"

SENS_CSV     = RESULTS_DIR / "layer_sensitivity.csv"
MEASURED_CSV = RESULTS_DIR / "layer_latency_measured.csv"
JETSON_CSV   = RESULTS_DIR / "layer_latency_jetson.csv"
PROXY_CSV    = RESULTS_DIR / "layer_latency_proxy.csv"
TRACES_CSV   = RESULTS_DIR / "hessian_traces.csv"

BUDGETS = {"tight": 0.60, "medium": 0.75, "relaxed": 0.90}
INT8_SPEEDUP = 0.50   # only used when no measured INT8 latency exists


# ── Hessian trace (Hutchinson) ────────────────────────────────────────────────

def make_loss_batch(model: YOLO, batch_size: int, device: str):
    """Yield preprocessed batches from the validation split for loss eval."""
    from ultralytics.cfg import get_cfg
    from ultralytics.data.build import build_yolo_dataset, build_dataloader
    from ultralytics.data.utils import check_det_dataset

    cfg = get_cfg()
    cfg.data = str(DATA_YAML)
    cfg.imgsz = 640
    data = check_det_dataset(str(DATA_YAML))
    ds = build_yolo_dataset(cfg, data["val"], batch=batch_size,
                            data=data, mode="val", rect=False)
    dl = build_dataloader(ds, batch=batch_size, workers=0, shuffle=True)
    for batch in dl:
        batch["img"] = batch["img"].to(device, non_blocking=True).float() / 255.0
        for k in ("batch_idx", "cls", "bboxes"):
            batch[k] = batch[k].to(device)
        yield batch


def hessian_traces(weights: str, probes: int, n_batches: int,
                   batch_size: int = 4) -> dict:
    """
    Hutchinson estimate of the per-layer Hessian trace of the detection loss,
    normalised by parameter count (HAWQ-V2 average trace), for every Conv2d.
    """
    from ultralytics.cfg import get_cfg

    device = "cuda"
    yolo = YOLO(weights)
    model = yolo.model.to(device).float()
    # Checkpoint-loaded models carry train args as a plain dict; the loss
    # criterion expects a namespace with box/cls/dfl gains.
    model.args = get_cfg()
    if hasattr(model, "criterion"):
        del model.criterion
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    conv_layers = [(n, m) for n, m in model.named_modules()
                   if isinstance(m, nn.Conv2d)]
    params = [m.weight for _, m in conv_layers]
    traces = {n: [] for n, _ in conv_layers}

    gen = make_loss_batch(yolo, batch_size, device)
    for b in range(n_batches):
        batch = next(gen)
        loss = model.loss(batch)[0].sum()
        # allow_unused: the DFL projection conv (model.22.dfl.conv) only
        # participates in the inference decode path, not the training loss,
        # so the Hessian proxy is structurally blind to it (trace := 0).
        grads = torch.autograd.grad(loss, params, create_graph=True,
                                    allow_unused=True)

        for p_idx, (name, _) in enumerate(conv_layers):
            g = grads[p_idx]
            if g is None:
                traces[name].append(0.0)
                continue
            layer_estimates = []
            for _ in range(probes):
                v = torch.randint_like(g, high=2, dtype=g.dtype) * 2 - 1
                hv = torch.autograd.grad((g * v).sum(), params[p_idx],
                                         retain_graph=True)[0]
                layer_estimates.append(float((hv * v).sum()))
            traces[name].append(np.mean(layer_estimates) / g.numel())
        del grads, loss
        torch.cuda.empty_cache()
        print(f"  Hessian batch {b + 1}/{n_batches} done")

    return {n: float(np.mean(v)) for n, v in traces.items()}


# ── Shared greedy assignment (same policy as 04_assignment.py) ───────────────

def load_latency() -> tuple:
    """Return (rows, kind) with per-layer FP16/INT8 latency."""
    if MEASURED_CSV.exists():
        with open(MEASURED_CSV, newline="") as f:
            rows = list(csv.DictReader(f))
        return ([{"layer_name": r["layer_name"],
                  "lat_fp16": float(r["lat_fp16_ms"]),
                  "lat_int8": float(r["lat_int8_ms"])} for r in rows],
                "trtexec_measured")
    src = JETSON_CSV if JETSON_CSV.exists() else PROXY_CSV
    with open(src, newline="") as f:
        rows = list(csv.DictReader(f))
    return ([{"layer_name": r["layer_name"],
              "lat_fp16": float(r["latency_ms"]),
              "lat_int8": float(r["latency_ms"]) * (1 - INT8_SPEEDUP)}
             for r in rows], src.stem)


def greedy_assign(sensitivity: dict, latency_rows: list, tag: str) -> None:
    total = sum(r["lat_fp16"] for r in latency_rows)
    for budget_name, frac in BUDGETS.items():
        remaining = total * frac
        assignment = {}
        ranked = sorted(
            latency_rows,
            key=lambda r: (sensitivity.get(r["layer_name"], 0.0)
                           / max(r["lat_fp16"] - r["lat_int8"], 1e-9)),
            reverse=True,
        )
        for r in ranked:
            if remaining >= r["lat_fp16"]:
                assignment[r["layer_name"]] = "FP16"
                remaining -= r["lat_fp16"]
            else:
                assignment[r["layer_name"]] = "INT8"
                remaining -= r["lat_int8"]

        out = RESULTS_DIR / f"assignment_{tag}_{budget_name}.json"
        with open(out, "w") as f:
            json.dump(assignment, f, indent=2)
        n16 = sum(1 for v in assignment.values() if v == "FP16")
        print(f"  [{tag}/{budget_name}] FP16={n16} INT8={len(assignment) - n16} → {out.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--probes", type=int, default=3)
    ap.add_argument("--batches", type=int, default=2)
    args = ap.parse_args()

    latency_rows, kind = load_latency()
    print(f"Latency source: {kind}")

    # 1. HAWQ-style Hessian-trace baseline
    if TRACES_CSV.exists():
        print(f"Reusing existing {TRACES_CSV}")
        with open(TRACES_CSV, newline="") as f:
            traces = {r["layer_name"]: float(r["avg_hessian_trace"])
                      for r in csv.DictReader(f)}
    else:
        print("Estimating per-layer Hessian traces (Hutchinson) …")
        traces = hessian_traces(args.weights, args.probes, args.batches)
        with open(TRACES_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["layer_name", "avg_hessian_trace"])
            w.writeheader()
            w.writerows([{"layer_name": n, "avg_hessian_trace": t}
                         for n, t in traces.items()])
        print(f"Saved {TRACES_CSV}")
    greedy_assign(traces, latency_rows, "hawq")

    # 2. FLOPs-proxy latency ablation (measured sensitivity, proxy latency)
    with open(SENS_CSV, newline="") as f:
        sens = {r["layer_name"]: float(r["delta_map"])
                for r in csv.DictReader(f)}
    with open(PROXY_CSV, newline="") as f:
        proxy_rows = [{"layer_name": r["layer_name"],
                       "lat_fp16": float(r["latency_ms"]),
                       "lat_int8": float(r["latency_ms"]) * (1 - INT8_SPEEDUP)}
                      for r in csv.DictReader(f)]
    greedy_assign(sens, proxy_rows, "flopsproxy")


if __name__ == "__main__":
    main()
