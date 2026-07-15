# Jetson Re-Run Runbook (paper revision)

Every step below regenerates data demanded by the reviewers:
measured per-layer INT8 latency (no more α=0.5), correctly-matched
mixed-precision engines (the ONNX→TRT name-mapping bug is fixed), full
validation-set mAP per engine, run-to-run latency variance, and the
HAWQ / FLOPs-proxy baseline engines.

## Step 0 — restore the baseline checkpoint (WORKSTATION, do this first)

The workstation copies of `runs/baseline/weights/best.pt` / `best.onnx`
were truncated to 0 bytes. The originals still live on the Jetson from the
May benchmark run. From the workstation:

```bash
scp jetson:<repo>/runs/baseline/weights/best.pt  runs/baseline/weights/best.pt
scp jetson:<repo>/runs/baseline/weights/best.onnx runs/baseline/weights/best.onnx
```

Sanity check (must print 64/64 bijective mapping for all three budgets):

```bash
python 05b_export_mixed.py --verify
```

If the checkpoint is gone on the Jetson too: retrain with
`python 01_train_baseline.py`, then re-run `02_sensitivity_analysis.py`
(the published sensitivity CSV is only valid for the original checkpoint).

## Step 1 — sync updated scripts + validation split to the Jetson

Copy to the Jetson repo:
- `03c_int8_latency_trtexec.py` (new)
- `04_assignment.py`, `05b_export_mixed.py`, `06_jetson_profile.sh`,
  `07_benchmark_jetson.py` (updated)
- `results/layer_sensitivity.csv` (unchanged, needed by 03c/04)
- the FULL validation split (2,501 images + labels) from
  `C:/Papers/Vision/People-Detection-(General)-8/valid/` — put it anywhere
  on the Jetson and point `data_jetson.yaml` at it:

```yaml
path: /home/<user>/datasets/people-detection
val: valid/images
nc: 1
names: [person]
```

Set MAXN power mode and lock clocks (recorded automatically by 07):

```bash
sudo nvpmodel -m 2        # MAXN SUPER (25 W) — check `nvpmodel -q` prints it
sudo jetson_clocks
```

## Step 2 — rebuild single-precision engines (if not already present)

```bash
python3 05_export.py      # FP32 / FP16 / INT8-PTQ engines + calibration set
```

## Step 3 — per-layer profiles (FP32 heatmap + FP16/INT8 for 03c)

```bash
./06_jetson_profile.sh
```

Produces `results/trtexec_profile_raw.txt`, `trtexec_profile_fp16.txt`,
`trtexec_profile_int8.txt`.

## Step 4 — measured INT8 latency + PyTorch-name FP16 latency

```bash
python3 03b_latency_jetson.py          # refresh layer_latency_jetson.csv
python3 03c_int8_latency_trtexec.py    # → layer_latency_measured.csv
```

03c prints how many of the 64 convs were directly measured; expect the
median measured α to be well above 0.5 — that is the paper's new
"realised speedup vs datasheet" data point.

## Step 5 — re-assign and build all mixed engines

```bash
python3 04_assignment.py               # now uses measured FP16+INT8 latency
python3 05b_export_mixed.py            # proposed: tight/medium/relaxed
```

05b now hard-fails unless all 64 Conv layers match, and writes
`results/engine_inspect_<budget>.json` — the per-layer realised-precision
proof for the paper. Keep these files.

For the baseline comparisons: copy `results/assignment_hawq_*.json` and
`results/assignment_flopsproxy_*.json` from the workstation (produced by
`04b_assignment_baselines.py` — see workstation section below), then re-run
`python3 05b_export_mixed.py` to build those six engines too.

## Step 6 — full benchmark

```bash
python3 07_benchmark_jetson.py
```

Now: 5 trtexec runs per engine (median ± std), engine-level mAP on the
full 2,501-image validation split (evaluates the .engine itself, not
best.pt — this fixes the identical-mAP bug in the old results_main.csv),
plus `results/jetson_env.txt` with nvpmodel/jetson_clocks state.

## Step 7 — copy results back to the workstation

```
results/layer_latency_jetson.csv
results/layer_latency_measured.csv
results/trtexec_profile_*.txt
results/assignment_*.json  results/assignment_summary.csv
results/engine_inspect_*.json
results/results_main.csv
results/jetson_env.txt
```

## Workstation follow-ups (after Step 0, can run in parallel with Jetson work)

```bash
python 02b_joint_sensitivity.py            # cross-layer coupling experiment
python 04b_assignment_baselines.py         # Hessian traces + baseline assignments
python 02c_sensitivity_second_dataset.py   # COCO-person cross-dataset check
python 08_plots.py                         # regenerate all figures
```

Note: `04b` needs latency CSVs; run it AFTER `layer_latency_measured.csv`
comes back from the Jetson so the baseline assignments use the same measured
latency as the proposed method (fair comparison). `02b` and `02c` have no
Jetson dependency.
