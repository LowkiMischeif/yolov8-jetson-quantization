"""
Phase 3c: Measured per-layer INT8 latency from trtexec profiles (Jetson).

Replaces the former fixed alpha = 0.5 INT8-speedup assumption with per-layer
*measured* FP16 and INT8 latencies, parsed from `trtexec --dumpProfile` runs
of the uniform FP16 and uniform INT8 engines:

    results/trtexec_profile_fp16.txt   (from 06_jetson_profile.sh)
    results/trtexec_profile_int8.txt   (from 06_jetson_profile.sh)

TensorRT fuses Conv+BN+activation into a single profiled layer whose name
still contains the ONNX Conv node path (e.g. '/model.0/conv/Conv + ...'),
so each profile row is attributed to the unique Conv2d whose ONNX name it
contains.  Convs whose row cannot be identified in one of the two profiles
(e.g. absorbed into a foreign fusion) fall back to the median measured
alpha of their group (head = 'model.22.*', otherwise backbone/neck), and
are flagged in the output.

Output: results/layer_latency_measured.csv
    layer_idx, layer_name, lat_fp16_ms, lat_int8_ms, alpha, source

Run ON THE JETSON after 06_jetson_profile.sh.
"""

import csv
import re
import statistics
from pathlib import Path

BASE        = Path(__file__).parent
RESULTS_DIR = BASE / "results"
ONNX_PATH   = BASE / "runs" / "baseline" / "weights" / "best.onnx"

FP16_PROFILE = RESULTS_DIR / "trtexec_profile_fp16.txt"
INT8_PROFILE = RESULTS_DIR / "trtexec_profile_int8.txt"
SENS_CSV     = RESULTS_DIR / "layer_sensitivity.csv"
OUT_CSV      = RESULTS_DIR / "layer_latency_measured.csv"

# Reuse the verified ONNX→PyTorch name normalisation from 05b.
import importlib.util
_spec = importlib.util.spec_from_file_location("m05b", BASE / "05b_export_mixed.py")
_m05b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m05b)
trt_conv_name_to_py_name = _m05b.trt_conv_name_to_py_name


def parse_trtexec_profile(path: Path) -> dict:
    """
    Parse the '=== Profile ===' table of a trtexec --dumpProfile log.

    Rows are '<layer name with possible spaces>  <time> <avg> <median> <pct>';
    the name is everything left of the trailing four numeric columns.
    Returns {trt_layer_name: median_time_ms}.
    """
    rows = {}
    in_table = False
    num_re = re.compile(
        r"^(.*?)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s+(-?\d+\.?\d*)\s*$"
    )
    for raw in path.read_text(errors="replace").splitlines():
        line = re.sub(r"^\[\S+\]\s+\[I\]\s?", "", raw)  # strip trtexec log prefix
        if "=== Profile" in line:
            in_table = True
            continue
        if not in_table:
            continue
        if line.strip().startswith(("Layer", "----")) or not line.strip():
            continue
        if line.strip().startswith(("Total", "===")):
            break
        m = num_re.match(line.rstrip())
        if m:
            name = m.group(1).strip()
            median_ms = float(m.group(4))
            rows[name] = median_ms
    if not rows:
        raise RuntimeError(f"No profile table found in {path}")
    return rows


def attribute_convs(profile: dict, onnx_conv_names: list) -> dict:
    """
    Map each of the 64 Conv ONNX node names to the profiled (possibly fused)
    TRT layer that contains it.  Returns {pytorch_name: latency_ms}.
    A profiled row containing several Conv names (rare) is split evenly.
    """
    row_to_convs = {row: [c for c in onnx_conv_names if c in row]
                    for row in profile}
    out = {}
    for row, convs in row_to_convs.items():
        if not convs:
            continue                       # Reformat / non-conv fused ops
        share = profile[row] / len(convs)
        for c in convs:
            py = trt_conv_name_to_py_name(c)
            out[py] = out.get(py, 0.0) + share
    return out


def main():
    import onnx

    graph = onnx.load(str(ONNX_PATH)).graph
    onnx_conv_names = [n.name for n in graph.node if n.op_type == "Conv"]
    print(f"ONNX Conv nodes: {len(onnx_conv_names)}")

    fp16 = attribute_convs(parse_trtexec_profile(FP16_PROFILE), onnx_conv_names)
    int8 = attribute_convs(parse_trtexec_profile(INT8_PROFILE), onnx_conv_names)
    print(f"Attributed convs — FP16 profile: {len(fp16)}, INT8 profile: {len(int8)}")

    # Layer order (layer_idx) comes from the sensitivity CSV.
    with open(SENS_CSV, newline="") as f:
        order = [(int(r["layer_idx"]), r["layer_name"])
                 for r in csv.DictReader(f)]

    measured_alphas = {
        name: int8[name] / fp16[name]
        for _, name in order
        if name in fp16 and name in int8 and fp16[name] > 1e-9
    }
    head_alphas = [a for n, a in measured_alphas.items() if n.startswith("model.22.")]
    body_alphas = [a for n, a in measured_alphas.items() if not n.startswith("model.22.")]
    fallback = {
        True:  statistics.median(head_alphas or list(measured_alphas.values()) or [1.0]),
        False: statistics.median(body_alphas or list(measured_alphas.values()) or [1.0]),
    }
    print(f"Measured alpha — median head: {fallback[True]:.3f}, "
          f"median backbone/neck: {fallback[False]:.3f} "
          f"({len(measured_alphas)}/{len(order)} layers directly measured)")

    rows = []
    for idx, name in order:
        lat_fp16 = fp16.get(name)
        lat_int8 = int8.get(name)
        if lat_fp16 is not None and lat_int8 is not None:
            alpha, source = lat_int8 / lat_fp16, "trtexec_measured"
        else:
            alpha  = fallback[name.startswith("model.22.")]
            source = "fallback_median_alpha"
            if lat_fp16 is None:
                # fall back to the PyTorch CUDA-event FP16 measurement
                lat_fp16 = _pytorch_fp16_latency().get(name, 0.0)
            lat_int8 = lat_fp16 * alpha
        rows.append({
            "layer_idx":   idx,
            "layer_name":  name,
            "lat_fp16_ms": round(lat_fp16, 6),
            "lat_int8_ms": round(lat_int8, 6),
            "alpha":       round(alpha, 4),
            "source":      source,
        })

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_meas = sum(1 for r in rows if r["source"] == "trtexec_measured")
    print(f"Saved {OUT_CSV}  ({n_meas}/{len(rows)} measured, "
          f"{len(rows) - n_meas} median-alpha fallback)")


def _pytorch_fp16_latency() -> dict:
    path = RESULTS_DIR / "layer_latency_jetson.csv"
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {r["layer_name"]: float(r["latency_ms"])
                for r in csv.DictReader(f)}


if __name__ == "__main__":
    main()
