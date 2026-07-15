"""
Phase 8: Generate all paper figures from results CSVs.

Run on any machine after experiments are complete.

Outputs (in results/figures/):
  pareto_plot.png          - mAP50 vs latency scatter (Pareto frontier)
  latency_heatmap.png      - per-layer latency contribution
  sensitivity_heatmap.png  - per-layer INT8 sensitivity (delta_mAP)
  assignment_map.png       - per-layer INT8/FP16 assignment grid (all budgets)
  quant_degradation.png    - mAP drop FP32 → FP16 → INT8 per condition
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":  "serif",
    "font.size":    11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi":   150,
})

CONDITION_COLORS = {
    "FP32":         "#2196F3",
    "FP16":         "#4CAF50",
    "INT8 PTQ":     "#FF5722",
    "Mixed tight":  "#9C27B0",
    "Mixed medium": "#673AB7",
    "Mixed relaxed":"#3F51B5",
    "Reference":    "#9E9E9E",
}


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ── Figure 1: Pareto frontier ─────────────────────────────────────────────────

def plot_pareto(results_csv: Path):
    if not results_csv.exists():
        print(f"SKIP pareto: {results_csv} not found.")
        return

    rows = load_csv(results_csv)
    rows = [r for r in rows if r["map50"] and r["latency_p50_ms"]]

    fig, ax = plt.subplots(figsize=(7, 5))

    for row in rows:
        label   = row["condition"]
        x       = float(row["latency_p50_ms"])
        y       = float(row["map50"]) * 100

        color = "#607D8B"
        for key, c in CONDITION_COLORS.items():
            if key.lower() in label.lower():
                color = c
                break

        marker = "o"
        if "mixed" in label.lower():
            marker = "D"
        elif "reference" in label.lower() or "teacher" in label.lower():
            marker = "^"

        xerr = float(row["latency_p50_std_ms"]) \
            if row.get("latency_p50_std_ms") else None
        ax.errorbar(x, y, xerr=xerr, fmt="none", ecolor=color,
                    elinewidth=1, capsize=2, zorder=4)
        ax.scatter(x, y, c=color, marker=marker, s=80, zorder=5)
        ax.annotate(
            label.replace("YOLOv8n ", "").replace("YOLOv8", "v8"),
            xy=(x, y), xytext=(5, 3), textcoords="offset points",
            fontsize=7, color="#333333",
        )

    # Draw Pareto frontier
    sorted_rows = sorted(rows, key=lambda r: float(r["latency_p50_ms"]))
    pareto, best_map = [], -1
    for r in sorted_rows:
        m = float(r["map50"])
        if m > best_map:
            pareto.append(r)
            best_map = m
    if len(pareto) > 1:
        px = [float(r["latency_p50_ms"]) for r in pareto]
        py = [float(r["map50"]) * 100      for r in pareto]
        ax.plot(px, py, "k--", linewidth=1, alpha=0.5, label="Pareto frontier")

    ax.set_xlabel("Inference Latency P50 (ms)")
    ax.set_ylabel("mAP50 (%)")
    ax.set_title("Accuracy–Latency Pareto Frontier\n(Jetson Orin Nano Super)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    out = FIGURES_DIR / "pareto_plot.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 2: Layer latency heatmap ──────────────────────────────────────────

def plot_latency_heatmap(latency_csv: Path):
    if not latency_csv.exists():
        print(f"SKIP latency heatmap: {latency_csv} not found.")
        return

    rows = load_csv(latency_csv)
    names     = [r["layer_name"] for r in rows]          # full dotted names
    latencies = [float(r["latency_ms"]) for r in rows]
    source    = rows[0].get("source", "") if rows else ""
    src_label = {"jetson_pytorch": "PyTorch CUDA-event timing",
                 "jetson_trtexec": "trtexec fused-layer profile",
                 "gpu_proxy":      "FLOPs proxy (workstation)"}.get(
                     source, source or "measured")

    top_n = 40
    if len(rows) > top_n:
        idx       = np.argsort(latencies)[::-1][:top_n]
        names     = [names[i] for i in idx]
        latencies = [latencies[i] for i in idx]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.22)))
    ax.barh(range(len(names)), latencies, color="#2196F3", alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=6.5, family="monospace")
    ax.invert_yaxis()                       # largest at top
    ax.set_xlabel("Per-layer latency (ms)")
    ax.set_ylabel("Conv2d layer (PyTorch module path)")
    ax.set_title("Top-40 Per-Layer Latency Contributions\n"
                 f"(Jetson Orin Nano Super, {src_label})")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    out = FIGURES_DIR / "latency_heatmap.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 3: Sensitivity heatmap ────────────────────────────────────────────

def plot_sensitivity_heatmap(sensitivity_csv: Path):
    if not sensitivity_csv.exists():
        print(f"SKIP sensitivity heatmap: {sensitivity_csv} not found.")
        return

    rows     = load_csv(sensitivity_csv)
    rows     = sorted(rows, key=lambda r: float(r["delta_map"]), reverse=True)[:40]
    names    = [r["layer_name"] for r in rows]           # full dotted names
    deltas   = [float(r["delta_map"]) * 100 for r in rows]

    colors = ["#FF5722" if d > 0.5 else "#FF9800" if d > 0.1 else "#4CAF50"
              for d in deltas]

    fig, ax = plt.subplots(figsize=(8, max(4, len(rows) * 0.22)))
    ax.barh(range(len(names)), deltas, color=colors, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=6.5, family="monospace")
    ax.invert_yaxis()                       # most sensitive at top
    ax.set_xlabel(r"Single-layer INT8 sensitivity $\Delta_i$ "
                  "(mAP50-95 drop, percentage points)")
    ax.set_ylabel("Conv2d layer (PyTorch module path)")
    ax.set_title("Per-Layer INT8 Sensitivity\n(Top 40 of 64 Conv2d layers)")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    legend_patches = [
        mpatches.Patch(color="#FF5722", label=r"High: $\Delta_i > 0.5$ pp"),
        mpatches.Patch(color="#FF9800", label=r"Medium: $0.1 < \Delta_i \leq 0.5$ pp"),
        mpatches.Patch(color="#4CAF50", label=r"Low: $\Delta_i \leq 0.1$ pp"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8)

    out = FIGURES_DIR / "sensitivity_heatmap.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 4: Assignment map ──────────────────────────────────────────────────

def plot_assignment_map(summary_csv: Path):
    if not summary_csv.exists():
        print(f"SKIP assignment map: {summary_csv} not found.")
        return

    rows    = load_csv(summary_csv)
    budgets = ["tight", "medium", "relaxed"]
    layers  = sorted(
        {r["layer_name"] for r in rows},
        key=lambda n: int([r["layer_idx"] for r in rows if r["layer_name"] == n][0])
    )

    grid = np.zeros((len(budgets), len(layers)))
    for r in rows:
        bi = budgets.index(r["budget"])
        li = layers.index(r["layer_name"])
        grid[bi, li] = 0 if r["precision"] == "INT8" else 1

    fig, ax = plt.subplots(figsize=(min(20, len(layers) * 0.25 + 2), 3))
    ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_yticks(range(len(budgets)))
    ax.set_yticklabels([f"{b.capitalize()} budget" for b in budgets])
    tick_step = max(1, len(layers) // 16)
    ax.set_xticks(range(0, len(layers), tick_step))
    ax.set_xticklabels(range(0, len(layers), tick_step), fontsize=7)
    ax.set_xlabel(f"Conv2d layer index (0 = input side, {len(layers) - 1} = "
                  "output side; detection head = model.22.*)")
    # Mark where the detection head begins
    head_start = next((i for i, n in enumerate(layers)
                       if n.startswith("model.22.")), None)
    if head_start is not None:
        ax.axvline(x=head_start - 0.5, color="black", linewidth=1.2,
                   linestyle=":", alpha=0.8)
        ax.text(head_start, -0.65, "detection head →", fontsize=7,
                ha="left", va="top")
    ax.set_title("Mixed-Precision Assignment Map\n(Green = FP16, Red = INT8)")

    legend_patches = [
        mpatches.Patch(color="#006837", label="FP16 (protected)"),
        mpatches.Patch(color="#a50026", label="INT8 (quantized)"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)

    out = FIGURES_DIR / "assignment_map.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 5: Quantization degradation ───────────────────────────────────────

def plot_quant_degradation(results_csv: Path):
    if not results_csv.exists():
        print(f"SKIP degradation: {results_csv} not found.")
        return

    rows = load_csv(results_csv)

    # Group by model family
    groups = {
        "YOLOv8n FP32":          None,
        "YOLOv8n FP16":          None,
        "YOLOv8n INT8 PTQ":      None,
        "YOLOv8n Mixed-PTQ medium": None,
    }
    for row in rows:
        cond = row["condition"]
        for key in groups:
            if key.lower() in cond.lower() and row["map50"]:
                groups[key] = float(row["map50"]) * 100

    labels = [k for k, v in groups.items() if v is not None]
    values = [groups[k] for k in labels]
    colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"][:len(labels)]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.85, width=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([l.replace("YOLOv8n ", "") for l in labels],
                       rotation=15, ha="right")
    ax.set_ylabel("mAP50 (%)")
    ax.set_title("Quantization Impact on Accuracy\n(YOLOv8n, People Detection)")
    ax.set_ylim(max(0, min(values) - 5), max(values) + 3)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    out = FIGURES_DIR / "quant_degradation.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 6: Joint-vs-additive sensitivity (cross-layer coupling) ───────────

def plot_joint_sensitivity(joint_csv: Path):
    if not joint_csv.exists():
        print(f"SKIP joint sensitivity: {joint_csv} not found.")
        return

    rows = load_csv(joint_csv)
    fig, ax = plt.subplots(figsize=(6, 5))

    n_budget = 0
    for r in rows:
        x = float(r["sum_single_delta"]) * 100
        y = float(r["joint_delta"]) * 100
        is_budget = r["config"].startswith("budget")
        ax.scatter(x, y, s=70, zorder=5,
                   c="#9C27B0" if is_budget else "#2196F3",
                   marker="s" if is_budget else "o")
        # Budget sets cluster near the origin — stagger their labels.
        offset = (8, 10 - 11 * n_budget) if is_budget else (5, 3)
        if is_budget:
            n_budget += 1
        ax.annotate(r["config"], xy=(x, y), xytext=offset,
                    textcoords="offset points", fontsize=7)

    lims = ax.get_xlim() + ax.get_ylim()
    lo, hi = min(lims), max(lims)
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.5,
            label="perfect additivity (y = x)")
    ax.set_xlabel(r"Sum of single-layer sensitivities "
                  r"$\sum_{i \in S}\Delta_i$ (pp mAP50-95)")
    ax.set_ylabel(r"Measured joint drop $\Delta_S$ (pp mAP50-95)")
    ax.set_title("Additivity of Per-Layer INT8 Sensitivity\n"
                 "(joint quantization of layer sets)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)

    out = FIGURES_DIR / "joint_sensitivity.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ── Figure 7: Cross-dataset sensitivity correlation ──────────────────────────

def plot_crossdataset(primary_csv: Path, coco_csv: Path):
    if not (primary_csv.exists() and coco_csv.exists()):
        print("SKIP cross-dataset plot: sensitivity CSVs incomplete.")
        return

    primary = {r["layer_name"]: float(r["delta_map"])
               for r in load_csv(primary_csv)}
    coco = {r["layer_name"]: float(r["delta_map"])
            for r in load_csv(coco_csv)}
    names = [n for n in primary if n in coco]
    xs = [primary[n] * 100 for n in names]
    ys = [coco[n] * 100 for n in names]
    head = [n.startswith("model.22.") for n in names]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter([x for x, h in zip(xs, head) if not h],
               [y for y, h in zip(ys, head) if not h],
               c="#4CAF50", s=40, alpha=0.8, label="backbone / neck", zorder=4)
    ax.scatter([x for x, h in zip(xs, head) if h],
               [y for y, h in zip(ys, head) if h],
               c="#FF5722", s=60, marker="D", label="detection head (model.22.*)",
               zorder=5)
    for n, x, y in zip(names, xs, ys):
        if abs(x) > 0.3 or abs(y) > 0.3:
            ax.annotate(n, xy=(x, y), xytext=(5, 3),
                        textcoords="offset points", fontsize=6.5)

    ax.set_xlabel(r"$\Delta_i$ on People Detection General v8 (pp mAP50-95)")
    ax.set_ylabel(r"$\Delta_i$ on COCO-person val2017 (pp mAP50-95)")
    ax.set_title("Per-Layer INT8 Sensitivity Across Datasets")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)

    out = FIGURES_DIR / "sensitivity_crossdataset.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Prefer trtexec fused-layer profile for the heatmap (real Jetson measurements).
    # Fall back to PyTorch-measured Jetson latency, then proxy.
    if (RESULTS_DIR / "trtexec_layer_profile.csv").exists():
        latency_csv = RESULTS_DIR / "trtexec_layer_profile.csv"
    elif (RESULTS_DIR / "layer_latency_jetson.csv").exists():
        latency_csv = RESULTS_DIR / "layer_latency_jetson.csv"
    else:
        latency_csv = RESULTS_DIR / "layer_latency_proxy.csv"

    plot_pareto(RESULTS_DIR / "results_main.csv")
    plot_latency_heatmap(latency_csv)
    plot_sensitivity_heatmap(RESULTS_DIR / "layer_sensitivity.csv")
    plot_assignment_map(RESULTS_DIR / "assignment_summary.csv")
    plot_quant_degradation(RESULTS_DIR / "results_main.csv")
    plot_joint_sensitivity(RESULTS_DIR / "joint_sensitivity.csv")
    plot_crossdataset(RESULTS_DIR / "layer_sensitivity.csv",
                      RESULTS_DIR / "layer_sensitivity_coco.csv")

    print(f"\nAll figures saved to {FIGURES_DIR}")
