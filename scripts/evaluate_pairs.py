#!/usr/bin/env python3
"""Evaluate enhanced images against paired bright references and write CSV."""
import argparse, csv
from pathlib import Path
from PIL import Image
from llie.metrics import image_metrics

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--predictions", required=True, help="Folder of enhanced images")
p.add_argument("--references", required=True, help="Folder of reference bright images with matching filenames")
p.add_argument("--output", default="reports/metrics.csv")
a = p.parse_args(); preds = Path(a.predictions); refs = Path(a.references); rows = []
for pred_path in sorted(preds.iterdir()):
    if not pred_path.is_file(): continue
    candidates = [refs / pred_path.name, refs / (pred_path.stem.removeprefix("enhanced_") + ".png"), refs / (pred_path.stem.removeprefix("enhanced_") + ".jpg")]
    ref_path = next((p for p in candidates if p.exists()), None)
    if ref_path is None: continue
    with Image.open(pred_path) as im, Image.open(ref_path) as ref:
        if im.size != ref.size: im = im.resize(ref.size)
        rows.append({"image": pred_path.name, **image_metrics(im, ref)})
if not rows: raise SystemExit("No matching prediction/reference pairs found.")
out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
print(f"Evaluated {len(rows)} image pairs. CSV saved to {out}")
