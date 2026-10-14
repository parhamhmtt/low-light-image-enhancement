#!/usr/bin/env python3
"""Apply a classical enhancement baseline to an image."""
import argparse
from pathlib import Path
import cv2
from llie.classical import METHODS

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--input", required=True); p.add_argument("--method", choices=sorted(METHODS), default="clahe")
p.add_argument("--output", default="outputs/classical.png")
a = p.parse_args()
image = cv2.imread(a.input, cv2.IMREAD_COLOR)
if image is None: raise SystemExit(f"Could not read input image: {a.input}")
result = METHODS[a.method](image); target = Path(a.output); target.parent.mkdir(parents=True, exist_ok=True)
if not cv2.imwrite(str(target), result): raise SystemExit(f"Could not save output: {target}")
print(f"Saved {target}")
