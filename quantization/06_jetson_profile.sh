#!/usr/bin/env bash
# Phase 6: Per-layer latency profiling on Jetson Orin Nano Super.
# Run this ON THE JETSON after 05_export.py has built the FP32 engine.
#
# Prerequisites on Jetson:
#   JetPack 6.x (includes TensorRT and trtexec)
#   trtexec is typically at /usr/src/tensorrt/bin/trtexec
#
# Usage:
#   chmod +x 06_jetson_profile.sh
#   ./06_jetson_profile.sh
#
# Output: results/trtexec_layer_profile.csv  (fused TRT layers, heatmap figure only)
#         Run 03b_latency_jetson.py for per-Conv2d latency used by 04_assignment.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE="${SCRIPT_DIR}/runs/engines/yolov8n_fp32.engine"
RESULTS_DIR="${SCRIPT_DIR}/results"
RAW_PROFILE="${RESULTS_DIR}/trtexec_profile_raw.txt"
# trtexec produces fused TRT layer names that don't match PyTorch module names.
# This CSV is used for the latency-heatmap figure only — NOT for 04_assignment.py.
# Run 03b_latency_jetson.py to get per-layer latency with matching PyTorch names.
OUT_CSV="${RESULTS_DIR}/trtexec_layer_profile.csv"
TRTEXEC="${TRTEXEC_PATH:-/usr/src/tensorrt/bin/trtexec}"

mkdir -p "${RESULTS_DIR}"

if [ ! -f "${ENGINE}" ]; then
    echo "ERROR: Engine not found at ${ENGINE}"
    echo "Run 05_export.py on the Jetson first."
    exit 1
fi

if [ ! -f "${TRTEXEC}" ]; then
    echo "ERROR: trtexec not found at ${TRTEXEC}"
    echo "Set TRTEXEC_PATH env var to the correct path."
    exit 1
fi

echo "=== Jetson Layer Latency Profiling ==="
echo "Engine  : ${ENGINE}"
echo "trtexec : ${TRTEXEC}"
echo ""

# Run trtexec with layer profiling
"${TRTEXEC}" \
    --loadEngine="${ENGINE}" \
    --warmUp=500 \
    --iterations=1000 \
    --dumpProfile \
    --separateProfileRun \
    --percentile=50,95 \
    2>&1 | tee "${RAW_PROFILE}"

echo ""
echo "Raw profile saved to ${RAW_PROFILE}"

# Parse the dumpProfile output into CSV (fused TRT layer names — for heatmap figure only)
# trtexec dumpProfile format:
#   [layer_name] median: X.XXX ms  time%: Y.YY%
python3 - <<'PYEOF'
import re
import csv
import sys
from pathlib import Path

raw_file = Path("results/trtexec_profile_raw.txt")
out_csv  = Path("results/trtexec_layer_profile.csv")

pattern = re.compile(
    r"(?P<name>.+?)\s+median:\s+(?P<median>[\d.]+)\s+ms.*?time%:\s+(?P<pct>[\d.]+)"
)

rows = []
with open(raw_file) as f:
    for i, line in enumerate(f):
        m = pattern.search(line)
        if m:
            rows.append({
                "layer_idx":    i,
                "layer_name":   m.group("name").strip(),
                "latency_ms":   float(m.group("median")),
                "time_pct":     float(m.group("pct")),
                "in_channels":  "",
                "out_channels": "",
                "kernel_size":  "",
                "input_h":      "",
                "input_w":      "",
                "source":       "jetson_trtexec",
            })

if not rows:
    print("WARNING: No layer profile lines parsed. Check trtexec output format.")
    sys.exit(1)

fieldnames = ["layer_idx", "layer_name", "in_channels", "out_channels",
              "kernel_size", "input_h", "input_w", "latency_ms", "source"]
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

total = sum(r["latency_ms"] for r in rows)
print(f"Parsed {len(rows)} layers.  Total measured latency: {total:.3f} ms")
print(f"TRT layer profile saved to {out_csv}")
print("NOTE: Use 03b_latency_jetson.py output (layer_latency_jetson.csv) for 04_assignment.py.")
PYEOF

echo ""
echo "=== Whole-model benchmark (all precision formats) ==="

for ENGINE_PATH in runs/engines/*.engine; do
    NAME=$(basename "${ENGINE_PATH}" .engine)
    echo ""
    echo "Benchmarking: ${NAME}"
    "${TRTEXEC}" \
        --loadEngine="${ENGINE_PATH}" \
        --warmUp=500 \
        --iterations=1000 \
        --percentile=50,95 \
        2>&1 | grep -E "mean|median|percentile|Throughput|GPU Memory"
done

echo ""
echo "Done."
echo "Next steps:"
echo "  python3 03b_latency_jetson.py   # per-layer latency with PyTorch names → layer_latency_jetson.csv"
echo "  python3 04_assignment.py         # greedy mixed-precision assignment"
echo "  python3 05b_export_mixed.py      # build mixed-precision TRT engines"
