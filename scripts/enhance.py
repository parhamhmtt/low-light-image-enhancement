#!/usr/bin/env python3
"""Enhance one image or a folder using a trained U-Net checkpoint."""
import argparse
from llie.inference import enhance_path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--input", required=True, help="Image path or directory")
p.add_argument("--checkpoint", default="checkpoints/best.pth")
p.add_argument("--output", default="outputs/enhanced")
p.add_argument("--device", default="auto", help="auto, cpu, cuda, mps")
p.add_argument("--base-channels", type=int, choices=(32, 64), default=None,
               help="Set explicitly if the checkpoint is a plain state_dict")
args = p.parse_args()
enhance_path(args.input, args.output, args.checkpoint, args.device, args.base_channels)
