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
    names    = [r["layer_name"].split(".")[-2] + "." + r["layer_name"].split(".")[-1]
                if "." in r["layer_name"] else r["layer_name"]
                for r in rows]
    latencies = [float(r["latency_ms"]) for r in rows]

    top_n = 40
    if len(rows) > top_n:
        idx      = np.argsort(latencies)[::-1][:top_n]
        names    = [names[i] for i in idx]
        latencies = [latencies[i] for i in idx]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.22)))
    bars = ax.barh(range(len(names)), latencies, color="#2196F3", alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Layer Latency (ms)")
    ax.set_title("Top Layer Latency Contributions\n(Jetson Orin Nano Super, FP32)")
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
    names    = [r["layer_name"].split(".")[-1] for r in rows]
    deltas   = [float(r["delta_map"]) * 100 for r in rows]

    colors = ["#FF5722" if d > 0.5 else "#FF9800" if d > 0.1 else "#4CAF50"
              for d in deltas]

    fig, ax = plt.subplots(figsize=(8, max(4, len(rows) * 0.22)))
    ax.barh(range(len(names)), deltas, color=colors, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("mAP50-95 Drop when Layer Quantized to INT8 (pp)")
    ax.set_title("Per-Layer INT8 Sensitivity Analysis\n(Top 40 most sensitive layers)")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    legend_patches = [
        mpatches.Patch(color="#FF5722", label="High sensitivity (>0.5 pp)"),
        mpatches.Patch(color="#FF9800", label="Medium (0.1–0.5 pp)"),
        mpatches.Patch(color="#4CAF50", label="Low (<0.1 pp)"),
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
    ax.set_xticks([])
    ax.set_xlabel(f"Layers (left=early, right=late)  [{len(layers)} total]")
    ax.set_title("Mixed-Precision Assignment Map\n(Green=FP16, Red=INT8)")

    legend_patches = [
        mpatches.Patch(color="#4CAF50", label="FP16"),
        mpatches.Patch(color="#FF5722", label="INT8"),
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

    print(f"\nAll figures saved to {FIGURES_DIR}")
