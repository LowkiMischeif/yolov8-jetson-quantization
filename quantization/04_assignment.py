"""
Phase 4: Greedy latency-budget-constrained mixed-precision assignment.

Algorithm:
  For each Conv2d layer i, compute a score:
      score_i = delta_map_i / latency_savings_i
  where latency_savings_i = latency_fp16_i - latency_int8_i  (estimated as 50% of fp16)

  Sort layers by score descending (highest accuracy-cost-per-ms-saved first).
  Assign FP16 to layers greedily until the latency budget is exhausted.
  Assign INT8 to all remaining layers.

Inputs:
  results/layer_sensitivity.csv   (from 02_sensitivity_analysis.py)
  results/layer_latency_jetson.csv  OR  results/layer_latency_proxy.csv

Outputs:
  results/assignment_tight.json    (budget = total * 0.60)
  results/assignment_medium.json   (budget = total * 0.75)
  results/assignment_relaxed.json  (budget = total * 0.90)
  results/assignment_summary.csv   (all layers with assigned precision per budget)
"""

import csv
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

SENSITIVITY_CSV    = RESULTS_DIR / "layer_sensitivity.csv"
JETSON_LATENCY_CSV = RESULTS_DIR / "layer_latency_jetson.csv"
PROXY_LATENCY_CSV  = RESULTS_DIR / "layer_latency_proxy.csv"

BUDGETS = {
    "tight":   0.60,
    "medium":  0.75,
    "relaxed": 0.90,
}

# INT8 layers run ~50% faster than FP16 on Jetson TensorRT (conservative estimate).
# Replace with measured values once Jetson profile is available.
INT8_SPEEDUP = 0.50


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    # Load sensitivity
    if not SENSITIVITY_CSV.exists():
        raise FileNotFoundError(
            f"{SENSITIVITY_CSV} not found. Run 02_sensitivity_analysis.py first."
        )
    sensitivity = {
        r["layer_name"]: float(r["delta_map"])
        for r in load_csv(SENSITIVITY_CSV)
    }

    # Load latency — prefer real Jetson measurements
    if JETSON_LATENCY_CSV.exists():
        latency_rows = load_csv(JETSON_LATENCY_CSV)
        source = "jetson"
        print("Using real Jetson latency profile.")
    elif PROXY_LATENCY_CSV.exists():
        latency_rows = load_csv(PROXY_LATENCY_CSV)
        source = "proxy"
        print("Using GPU proxy latency (replace with Jetson profile for final paper).")
    else:
        raise FileNotFoundError(
            "No latency CSV found. Run 03_latency_proxy.py or 06_jetson_profile.sh first."
        )

    # Merge sensitivity and latency on layer_name
    layers = []
    for row in latency_rows:
        name = row["layer_name"]
        lat_fp16 = float(row["latency_ms"])
        lat_int8 = lat_fp16 * (1 - INT8_SPEEDUP)
        savings  = lat_fp16 - lat_int8
        sens     = sensitivity.get(name, 0.0)

        # Score: accuracy cost per ms saved by going INT8.
        # Higher score → more costly to quantize → assign FP16 preferentially.
        score = sens / savings if savings > 1e-9 else float("inf")

        layers.append({
            "layer_name":  name,
            "layer_idx":   int(row["layer_idx"]),
            "lat_fp16_ms": lat_fp16,
            "lat_int8_ms": lat_int8,
            "savings_ms":  savings,
            "delta_map":   sens,
            "score":       score,
            "source":      source,
        })

    total_fp16_latency = sum(l["lat_fp16_ms"] for l in layers)
    print(f"\nTotal FP16 latency (all layers): {total_fp16_latency:.3f} ms")
    print(f"Layers: {len(layers)}\n")

    # Sort by score descending — highest accuracy-cost first
    sorted_layers = sorted(layers, key=lambda l: l["score"], reverse=True)

    summary_rows = []

    for budget_name, budget_frac in BUDGETS.items():
        target_latency = total_fp16_latency * budget_frac
        remaining      = target_latency
        assignment     = {}

        for layer in sorted_layers:
            name = layer["layer_name"]
            if remaining >= layer["lat_fp16_ms"]:
                # Afford FP16 — keep high precision
                assignment[name] = "FP16"
                remaining -= layer["lat_fp16_ms"]
            else:
                # Must use INT8 to stay within budget
                assignment[name] = "INT8"
                remaining -= layer["lat_int8_ms"]

        fp16_count   = sum(1 for v in assignment.values() if v == "FP16")
        int8_count   = sum(1 for v in assignment.values() if v == "INT8")
        est_latency  = sum(
            l["lat_fp16_ms"] if assignment[l["layer_name"]] == "FP16"
            else l["lat_int8_ms"]
            for l in layers
        )
        est_map_loss = sum(
            l["delta_map"] for l in layers
            if assignment[l["layer_name"]] == "INT8"
        )

        out_path = RESULTS_DIR / f"assignment_{budget_name}.json"
        with open(out_path, "w") as f:
            json.dump(assignment, f, indent=2)

        print(f"Budget [{budget_name}] {budget_frac*100:.0f}% of FP16 latency "
              f"({target_latency:.2f} ms)")
        print(f"  FP16 layers : {fp16_count}")
        print(f"  INT8 layers : {int8_count}")
        print(f"  Est. latency: {est_latency:.2f} ms")
        print(f"  Est. mAP loss (cumulative delta): {est_map_loss:.4f}")
        print(f"  Saved to    : {out_path}\n")

        for layer in layers:
            summary_rows.append({
                "budget":      budget_name,
                "layer_idx":   layer["layer_idx"],
                "layer_name":  layer["layer_name"],
                "precision":   assignment[layer["layer_name"]],
                "lat_fp16_ms": round(layer["lat_fp16_ms"], 6),
                "lat_int8_ms": round(layer["lat_int8_ms"], 6),
                "delta_map":   round(layer["delta_map"], 6),
                "score":       round(layer["score"], 6) if layer["score"] != float("inf") else 999,
            })

    # Write full summary CSV
    summary_csv = RESULTS_DIR / "assignment_summary.csv"
    fieldnames  = ["budget", "layer_idx", "layer_name", "precision",
                   "lat_fp16_ms", "lat_int8_ms", "delta_map", "score"]
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Full summary saved to {summary_csv}")


if __name__ == "__main__":
    main()
