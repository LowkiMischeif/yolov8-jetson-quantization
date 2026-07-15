"""
Sequential runner for all workstation-side revision experiments.

Order matters:
  1. 02_sensitivity_analysis.py   — regenerate per-layer deltas in the current
                                    environment (baseline drifted ~0.5pp since
                                    the May run; keep everything consistent)
  2. 02b_joint_sensitivity.py     — joint sets vs sum of singles
  3. 02c_sensitivity_second_dataset.py — COCO-person cross-dataset check
  4. 04b_assignment_baselines.py  — Hessian traces (+ provisional assignments;
                                    re-run after measured Jetson latency lands)
  5. 08_plots.py                  — regenerate all figures

Usage: python run_workstation_experiments.py
"""

import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).parent
STEPS = [
    "02_sensitivity_analysis.py",
    "02b_joint_sensitivity.py",
    "02c_sensitivity_second_dataset.py",
    "04b_assignment_baselines.py",
    "08_plots.py",
]

if __name__ == "__main__":
    for step in STEPS:
        t0 = time.time()
        print(f"### STEP START {step}", flush=True)
        rc = subprocess.run([sys.executable, str(BASE / step)],
                            cwd=str(BASE)).returncode
        mins = (time.time() - t0) / 60
        status = "OK" if rc == 0 else f"FAILED rc={rc}"
        print(f"### STEP END {step} {status} ({mins:.1f} min)", flush=True)
        if rc != 0:
            print("### RUN ABORTED", flush=True)
            sys.exit(rc)
    print("### RUN COMPLETE", flush=True)
