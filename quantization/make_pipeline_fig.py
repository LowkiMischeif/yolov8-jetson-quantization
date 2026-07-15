"""
Generate the four-phase pipeline diagram (Fig. 1 of the paper).

Output: results/figures/pipeline.png
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIGURES_DIR = Path(__file__).parent / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

WORK_C  = "#E3F2FD"   # workstation phases
JET_C   = "#E8F5E9"   # Jetson phases
EDGE_W  = "#1565C0"
EDGE_J  = "#2E7D32"


def box(ax, x, y, w, h, title, lines, fc, ec):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02",
        facecolor=fc, edgecolor=ec, linewidth=1.5))
    n_title = title.count("\n") + 1
    ax.text(x + w / 2, y + h - 0.08, title, ha="center", va="top",
            fontsize=10, fontweight="bold")
    ax.text(x + w / 2, y + h - 0.10 - 0.30 * n_title, "\n".join(lines),
            ha="center", va="top", fontsize=8.2)


def arrow(ax, x1, y1, x2, y2, label=None):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
        linewidth=1.4, color="#37474F"))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.04, label,
                ha="center", fontsize=7.5, style="italic", color="#37474F")


def main():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    bw, bh, ty = 2.05, 1.55, 2.5

    box(ax, 0.25, ty, bw, bh, "Phase 1\nBaseline Training",
        ["YOLOv8n fine-tune", "People Detection v8", "20,598 images",
         "→ best.pt (FP32)"], WORK_C, EDGE_W)
    box(ax, 2.75, ty, bw, bh, "Phase 2\nSensitivity Analysis",
        ["One-at-a-time INT8", "64 Conv2d layers",
         r"→ $\Delta_i$ per layer", "+ joint-set validation"],
        WORK_C, EDGE_W)
    box(ax, 5.25, ty, bw, bh, "Phase 3\nLatency Profiling",
        ["On-device (Jetson)", "FP16 + INT8 engines",
         r"→ $L_i^{\mathrm{FP16}}, L_i^{\mathrm{INT8}}$",
         "trtexec per-layer"], JET_C, EDGE_J)
    box(ax, 7.75, ty, bw, bh, "Phase 4\nAssignment + Build",
        ["Greedy knapsack", "budget $B$",
         "→ per-layer INT8/FP16", "TensorRT engine"], JET_C, EDGE_J)

    arrow(ax, 0.25 + bw, ty + bh / 2, 2.75, ty + bh / 2)
    arrow(ax, 2.75 + bw, ty + bh / 2, 5.25, ty + bh / 2,
          r"$\boldsymbol{\Delta}$")
    arrow(ax, 5.25 + bw, ty + bh / 2, 7.75, ty + bh / 2,
          r"$\mathbf{L}$")

    # Bottom row: platform annotation + outputs
    box(ax, 0.25, 0.55, 4.55, 1.1, "GPU Workstation (RTX 4070)",
        ["Phases 1–2 run once per model/dataset",
         "sensitivity vector reusable across devices"], WORK_C, EDGE_W)
    box(ax, 5.25, 0.55, 4.55, 1.1, "NVIDIA Jetson Orin Nano Super",
        ["Phases 3–4 repeated per device / budget",
         "engine verified with IEngineInspector"], JET_C, EDGE_J)

    arrow(ax, 2.5, 1.65, 2.5, ty - 0.02)
    arrow(ax, 7.5, 1.65, 7.5, ty - 0.02)

    fig.tight_layout()
    out = FIGURES_DIR / "pipeline.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
