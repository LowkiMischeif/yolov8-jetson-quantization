"""
Regenerate pareto_plot.png — uses arrow annotations for clustered points.
Run: python regenerate_pareto.py
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = [
    {"label": "FP32",              "lat": 6.88, "map50": 88.5, "marker": "s", "color": "#2196F3"},
    {"label": "FP16",              "lat": 3.79, "map50": 88.5, "marker": "s", "color": "#4CAF50"},
    {"label": "INT8 PTQ (uniform)","lat": 3.25, "map50": 88.5, "marker": "o", "color": "#FF5722"},
    {"label": "Mixed tight",       "lat": 3.24, "map50": 88.5, "marker": "D", "color": "#9C27B0"},
    {"label": "Mixed medium",      "lat": 3.17, "map50": 88.5, "marker": "D", "color": "#673AB7"},
    {"label": "Mixed relaxed",     "lat": 3.17, "map50": 88.5, "marker": "D", "color": "#3F51B5"},
]

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.labelsize": 9, "figure.dpi": 220,
})

fig, ax = plt.subplots(figsize=(5.2, 3.8))

# Plot all points
for c in CONDITIONS:
    ax.scatter(c["lat"], c["map50"],
               color=c["color"], marker=c["marker"],
               s=80, zorder=5)

# ── Simple labels for well-separated points ──────────────────────────────────
ax.annotate("FP32",  xy=(6.88, 88.5), xytext=(6.5, 88.65),
            fontsize=7.5, color="#2196F3",
            arrowprops=dict(arrowstyle="-", color="#2196F3", lw=0.8))

ax.annotate("FP16",  xy=(3.79, 88.5), xytext=(4.1, 88.72),
            fontsize=7.5, color="#4CAF50",
            arrowprops=dict(arrowstyle="-", color="#4CAF50", lw=0.8))

# ── Arrow annotations for the clustered group — text placed above ─────────────
cluster_labels = [
    ("INT8 PTQ\n(uniform)", 3.25, 88.5, "#FF5722", (3.55, 89.55)),
    ("Mixed\ntight",        3.24, 88.5, "#9C27B0", (3.30, 89.80)),
    ("Mixed\nmedium",       3.17, 88.5, "#673AB7", (2.80, 89.80)),
    ("Mixed\nrelaxed",      3.17, 88.5, "#3F51B5", (2.80, 89.35)),
]

for label, x, y, color, (tx, ty) in cluster_labels:
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(tx, ty),
        fontsize=7,
        color=color,
        ha="center",
        va="bottom",
        multialignment="center",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color,
                  lw=0.8, alpha=0.9),
        arrowprops=dict(
            arrowstyle="->",
            color=color,
            lw=0.9,
            connectionstyle="arc3,rad=0.1",
        ),
    )

# ── 2.5% latency improvement arrow ───────────────────────────────────────────
ax.annotate("",
            xy=(3.17, 88.35), xytext=(3.25, 88.35),
            arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.0))
ax.text(3.21, 88.27, "−2.5% lat.", fontsize=6.5,
        ha="center", color="#555555")

ax.set_xlabel("Inference Latency P50 (ms)")
ax.set_ylabel("mAP50 (%)")
ax.set_title("Accuracy–Latency Trade-off\n(Jetson Orin Nano Super)")
ax.set_ylim(87.8, 90.4)
ax.set_xlim(2.3, 7.8)
ax.grid(True, linestyle="--", alpha=0.35)

# Legend via patches only
patches = [mpatches.Patch(color=c["color"], label=c["label"])
           for c in CONDITIONS]
ax.legend(handles=patches, loc="lower right", fontsize=6.8,
          ncol=2, framealpha=0.92, edgecolor="#cccccc",
          handlelength=1.0, handleheight=0.8)

fig.tight_layout()
out = FIGURES_DIR / "pareto_plot.png"
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"Saved {out}")
