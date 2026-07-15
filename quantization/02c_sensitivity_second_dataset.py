"""
Phase 2c: Cross-dataset check of the per-layer sensitivity pattern.

Repeats the one-at-a-time INT8 sensitivity analysis (02_sensitivity_analysis.py)
on a SECOND person-detection dataset — the person-only subset of COCO val2017 —
and reports the Spearman rank correlation between the two sensitivity vectors.
A high correlation shows the DFL-head-dominates finding is a property of the
architecture, not of the primary dataset.

Steps (all workstation-side):
  1. Extract datasets/coco_person/val2017.zip + annotations.zip
     (downloaded from images.cocodataset.org).
  2. Keep only images containing >= 1 'person' annotation; convert those
     annotations to YOLO format with a single class 0 = person.
  3. Baseline val + 64 single-layer INT8 vals on the COCO-person subset.

Outputs:
  results/layer_sensitivity_coco.csv    (same schema as layer_sensitivity.csv)
  printed Spearman/Pearson correlation vs the primary dataset

Usage:
    python 02c_sensitivity_second_dataset.py [--weights PATH] [--limit-layers N]
"""

import argparse
import csv
import json
import zipfile
from pathlib import Path

import torch.nn as nn
from ultralytics import YOLO

BASE        = Path(__file__).parent
WEIGHTS     = BASE / "runs" / "baseline" / "weights" / "best.pt"
RESULTS_DIR = BASE / "results"
DS_DIR      = BASE / "datasets" / "coco_person"
PRIMARY_CSV = RESULTS_DIR / "layer_sensitivity.csv"
OUT_CSV     = RESULTS_DIR / "layer_sensitivity_coco.csv"

import importlib.util
_spec = importlib.util.spec_from_file_location("m02", BASE / "02_sensitivity_analysis.py")
_m02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m02)
fake_int8 = _m02.fake_int8


# ── Dataset preparation ───────────────────────────────────────────────────────

def prepare_dataset() -> Path:
    """Extract COCO val2017, filter to person images, convert to YOLO layout."""
    yaml_path = DS_DIR / "data.yaml"
    img_out   = DS_DIR / "images" / "val"
    lbl_out   = DS_DIR / "labels" / "val"
    if yaml_path.exists() and any(img_out.glob("*.jpg")):
        print(f"COCO-person subset already prepared at {DS_DIR}")
        return yaml_path

    ann_zip = DS_DIR / "annotations.zip"
    img_zip = DS_DIR / "val2017.zip"
    for z in (ann_zip, img_zip):
        if not z.exists():
            raise FileNotFoundError(
                f"{z} missing — download val2017.zip and "
                "annotations_trainval2017.zip from images.cocodataset.org first."
            )

    print("Reading COCO annotations …")
    with zipfile.ZipFile(ann_zip) as zf:
        with zf.open("annotations/instances_val2017.json") as f:
            coco = json.load(f)

    person_cat = next(c["id"] for c in coco["categories"] if c["name"] == "person")
    images = {im["id"]: im for im in coco["images"]}
    boxes_by_img: dict = {}
    for ann in coco["annotations"]:
        if ann["category_id"] != person_cat or ann.get("iscrowd", 0):
            continue
        boxes_by_img.setdefault(ann["image_id"], []).append(ann["bbox"])
    print(f"Images with >=1 person: {len(boxes_by_img)} / {len(images)}")

    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    wanted = {images[i]["file_name"]: i for i in boxes_by_img}
    print("Extracting person images …")
    with zipfile.ZipFile(img_zip) as zf:
        for member in zf.namelist():
            fname = Path(member).name
            if fname in wanted:
                with zf.open(member) as src, open(img_out / fname, "wb") as dst:
                    dst.write(src.read())

    print("Writing YOLO labels …")
    for fname, img_id in wanted.items():
        im = images[img_id]
        w, h = im["width"], im["height"]
        lines = []
        for x, y, bw, bh in boxes_by_img[img_id]:
            cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
            lines.append(f"0 {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
        (lbl_out / (Path(fname).stem + ".txt")).write_text("\n".join(lines))

    yaml_path.write_text(
        f"path: {DS_DIR.as_posix()}\n"
        "train: images/val\n"   # unused; val-only dataset
        "val: images/val\n"
        "nc: 1\n"
        "names:\n  - person\n"
    )
    print(f"Prepared {len(wanted)} images → {yaml_path}")
    return yaml_path


# ── Sensitivity sweep ─────────────────────────────────────────────────────────

def run_val(model: YOLO, data_yaml: Path) -> float:
    metrics = model.val(data=str(data_yaml), imgsz=640, batch=16,
                        device=0, verbose=False)
    return float(metrics.box.map)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--limit-layers", type=int, default=0,
                    help="smoke test: only analyse the first N layers")
    args = ap.parse_args()

    data_yaml = prepare_dataset()

    model = YOLO(args.weights)
    conv_layers = [(n, m) for n, m in model.model.named_modules()
                   if isinstance(m, nn.Conv2d)]
    print(f"{len(conv_layers)} Conv2d layers")

    print("Baseline val on COCO-person …")
    baseline = run_val(model, data_yaml)
    print(f"Baseline mAP50-95 (COCO-person): {baseline:.5f}\n")

    todo = conv_layers[:args.limit_layers] if args.limit_layers else conv_layers
    rows = []
    for i, (name, module) in enumerate(todo):
        hook = module.register_forward_hook(
            lambda m, inp, out: fake_int8(out.detach()))
        try:
            quant_map = run_val(model, data_yaml)
        finally:
            hook.remove()
        delta = baseline - quant_map
        rows.append({
            "layer_idx":    i,
            "layer_name":   name,
            "in_channels":  module.in_channels,
            "out_channels": module.out_channels,
            "kernel_size":  module.kernel_size[0],
            "baseline_map": round(baseline, 6),
            "quant_map":    round(quant_map, 6),
            "delta_map":    round(delta, 6),
        })
        print(f"[{i + 1:3d}/{len(todo)}] {name:<50s} delta={delta:+.5f}")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {OUT_CSV}")

    if not args.limit_layers and PRIMARY_CSV.exists():
        with open(PRIMARY_CSV, newline="") as f:
            primary = {r["layer_name"]: float(r["delta_map"])
                       for r in csv.DictReader(f)}
        names = [r["layer_name"] for r in rows if r["layer_name"] in primary]
        a = [primary[n] for n in names]
        b = [next(r["delta_map"] for r in rows if r["layer_name"] == n)
             for n in names]
        try:
            from scipy.stats import spearmanr, pearsonr
            s, sp = spearmanr(a, b)
            p, pp = pearsonr(a, b)
            print(f"Spearman rho = {s:.3f} (p={sp:.2e}); "
                  f"Pearson r = {p:.3f} (p={pp:.2e})")
        except ImportError:
            import numpy as np
            ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
            s = np.corrcoef(ra, rb)[0, 1]
            print(f"Spearman rho (numpy) = {s:.3f}")


if __name__ == "__main__":
    main()
