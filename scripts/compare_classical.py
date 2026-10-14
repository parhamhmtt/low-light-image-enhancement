#!/usr/bin/env python3
"""Run every classical baseline on one image and save a comparison panel."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from llie.classical import METHODS

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--input", required=True)
p.add_argument("--output-dir", default="outputs/classical_comparison")
p.add_argument("--max-width", type=int, default=480)
a = p.parse_args()
image = cv2.imread(a.input, cv2.IMREAD_COLOR)
if image is None: raise SystemExit(f"Could not read input image: {a.input}")
out_dir = Path(a.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
h, w = image.shape[:2]; scale = min(1.0, a.max_width / w)
thumb_size = (max(1, int(w * scale)), max(1, int(h * scale)))
items = [("Original", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))]
for name, method in METHODS.items():
    result = method(image); cv2.imwrite(str(out_dir / f"{name}.png"), result)
    items.append((name.upper(), cv2.cvtColor(result, cv2.COLOR_BGR2RGB)))
label_h = 28; columns = 2; rows = (len(items) + columns - 1) // columns
canvas = Image.new("RGB", (columns * thumb_size[0], rows * (thumb_size[1] + label_h)), "white")
draw = ImageDraw.Draw(canvas)
for i, (label, arr) in enumerate(items):
    thumb = Image.fromarray(arr).resize(thumb_size)
    x = (i % columns) * thumb_size[0]; y = (i // columns) * (thumb_size[1] + label_h)
    canvas.paste(thumb, (x, y + label_h)); draw.text((x + 8, y + 7), label, fill="black")
panel = out_dir / "comparison.png"; canvas.save(panel)
print(f"Saved all methods and comparison panel to {out_dir.resolve()} (panel: {panel.name})")
