# Latency-Profile-Guided Mixed-Precision Quantization for Real-Time Person Detection on the NVIDIA Jetson Orin Nano Super

Code and results for a latency-profile-guided **mixed-precision quantization** framework for YOLOv8 person detection on the NVIDIA Jetson Orin Nano Super. Instead of uniform INT8 post-training quantization (PTQ), each `Conv2d` layer is assigned INT8 or FP16 using two empirical signals — per-layer on-device latency and per-layer mAP sensitivity to INT8 — combined via a greedy knapsack assignment under a target latency budget.

## Dataset

Trained and evaluated on the **People-Detection (General) 8** dataset — a single-class (`person`) detection dataset of ~20,598 training and ~2,501 validation images at 640×640. See [`quantization/data.yaml`](quantization/data.yaml) for the train/val/test split configuration.

> The dataset itself is **not** included in this repository (see `.gitignore`); point `data.yaml` at your local copy to reproduce.

## Pipeline

The numbered scripts run end to end:

| Step | Script | Purpose |
|------|--------|---------|
| 01 | `01_train_baseline.py` | Fine-tune the YOLOv8n baseline |
| 02 | `02_sensitivity_analysis.py` | Per-layer INT8 mAP sensitivity |
| 03 | `03_latency_proxy.py` / `03b_latency_jetson.py` | Per-layer latency (proxy + on-device) |
| 04 | `04_assignment.py` | Greedy precision assignment under latency budgets |
| 05 | `05_export.py` / `05b_export_mixed.py` | Export INT8 / mixed-precision engines |
| 06 | `06_jetson_profile.sh` | `trtexec` per-layer profiling on Jetson |
| 07 | `07_benchmark_jetson.py` | End-to-end benchmark on Jetson |
| 08 | `08_plots.py` / `regenerate_pareto.py` | Figures and Pareto plots |

## Results

Precomputed metrics and figures are tracked under [`quantization/results/`](quantization/results/):

- `results_main.csv`, `assignment_summary.csv` — headline metrics
- `layer_sensitivity.csv`, `layer_latency_jetson.csv`, `layer_latency_proxy.csv` — per-layer measurements
- `assignment_{tight,medium,relaxed}.json` — precision maps at three latency budgets
- `figures/` — sensitivity/latency heatmaps, assignment map, Pareto plot

The full write-up is included as the PDF in `quantization/`.

## Setup

```bash
pip install -r quantization/requirements.txt
```

## License

MIT — see [`quantization/LICENSE`](quantization/LICENSE).
