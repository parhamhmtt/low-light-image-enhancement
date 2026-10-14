# Low-Light Image Enhancement

## Project pipeline

![Low-light image enhancement pipeline](assets/pipeline-pic.png)

# Low-Light Image Enhancement

A reproducible computer-vision project for enhancing low-light photographs using classical image-processing methods and a residual U-Net implemented in PyTorch.

The project is based on the supplied training code and research documents. It provides a clean package structure, training and inference scripts, classical baselines, and evaluation utilities.

> **Pre-trained checkpoint:** Download `best.pth` and any other required resources from the repository's `checkpoints/` directory. Place the checkpoint at `checkpoints/best.pth`. The trained checkpoint is included in this repository. Check its file size and Git LFS configuration if you plan to update or replace it. Compatibility must be confirmed against the exact architecture and checkpoint format.

## Pipeline

### Stage 1 — Data preparation
- Organize aligned dark/bright image pairs into training and validation splits.
- Load RGB images and convert them to tensors in the `[0, 1]` range.
- Apply synchronized random crops, horizontal/vertical flips, and 90-degree rotations to both images in a pair.
- Keep paired filenames aligned so each low-light image has its corresponding normal-light target.

### Stage 2 — Model training
- Train a residual U-Net encoder–decoder with skip connections and residual blocks.
- Use an L1 reconstruction loss, AdamW optimizer, cosine learning-rate scheduling, and gradient clipping.
- Track training/validation loss and validation PSNR.
- Save `best.pth`, `last.pth`, and training history when training is run locally.

### Stage 3 — Inference and evaluation
- Load a checkpoint and enhance one image or a directory of images.
- Save enhanced images in an output directory.
- Compare classical enhancement methods.
- When paired reference images are available, calculate PSNR, SSIM, and MAE.

## Methods

Classical methods included in the repository:
- Histogram Equalization
- CLAHE
- Gamma Correction
- Single-Scale Retinex (SSR)
- Multi-Scale Retinex (MSR)
- Multi-Scale Retinex with Color Restoration (MSRCR)
- Exposure Fusion

Deep-learning method:
- Residual U-Net with an encoder, bottleneck, decoder, skip connections, and sigmoid RGB output.

## Repository structure

```text
low-light-image-enhancement/
├── src/
│   └── llie/
│       ├── __init__.py
│       ├── model.py          # Residual U-Net architecture
│       ├── data.py           # Paired dark/bright dataset loader
│       ├── train.py          # Training loop and checkpoint saving
│       ├── inference.py      # Checkpoint loading and inference
│       ├── classical.py      # Classical enhancement methods
│       └── metrics.py        # PSNR, SSIM, and MAE
├── scripts/
│   ├── enhance.py
│   ├── enhance_classical.py
│   ├── compare_classical.py
│   └── evaluate_pairs.py
├── tests/
├── docs/                     # Supplied research paper (PDF and DOCX)
├── reports/
│   └── experiment_summary.md # Transcribed paper results, not a fresh rerun
├── assets/
├── requirements.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

## Why is the package inside `src/llie/`?

`llie` is a short package name for **L**ow-**L**ight **I**mage **E**nhancement. It is not a typo for `illo`.

The `src/` layout is a common Python packaging convention: importable project code lives inside `src/llie/`, while command-line entry points, tests, documentation, and configuration stay at the repository root. The package is installed in editable mode with `pip install -e .`, which makes `import llie` work during development and testing.

Do not move `model.py`, `train.py`, or the other package modules out of `src/llie/` unless you also update the package configuration and imports.

## Installation

Python 3.10 or newer is recommended. Run these commands from the repository root.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

For GPU acceleration, install the PyTorch build compatible with your CUDA version using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

## Quick start: enhance an image

1. The trained checkpoint is included in the repository in `checkpoints/best.pth`.
3. Run:

```bash
python scripts/enhance.py --input path/to/dark_photo.jpg --checkpoint checkpoints/best.pth --output outputs/enhanced
```

To process a directory:

```bash
python scripts/enhance.py --input path/to/dark_images --checkpoint checkpoints/best.pth --output outputs/enhanced
```

If needed, select the device explicitly:

```bash
python scripts/enhance.py --input path/to/dark_photo.jpg --checkpoint checkpoints/best.pth --output outputs/enhanced --device cpu
```

Check `python scripts/enhance.py --help` for the exact options supported by the checked-out version.

## Classical enhancement

Run CLAHE on one image:

```bash
python scripts/enhance_classical.py --input path/to/dark_photo.jpg --method clahe --output outputs/clahe.png
```

Compare all classical methods and create a comparison panel:

```bash
python scripts/compare_classical.py --input path/to/dark_photo.jpg --output-dir outputs/classical_comparison
```

Supported method names are `he`, `clahe`, `gamma`, `ssr`, `msr`, `msrcr`, and `fusion`.

## Dataset format for training

The training loader expects matching filenames for paired low-light and bright images:

```text
data/paired/
├── train/
│   ├── dark/
│   │   ├── image_001.png
│   │   └── ...
│   └── bright/
│       ├── image_001.png
│       └── ...
└── val/
    ├── dark/
    └── bright/
```

Each dark image must have a corresponding bright target with the same filename. Keep the images aligned and use a dataset whose license permits your intended use. The dataset itself is not included in this repository.

Start training:

```bash
python -m llie.train --data data/paired --epochs 100 --batch-size 4 --patch-size 256 --base-channels 32 --output checkpoints
```

Resume from the last checkpoint:

```bash
python -m llie.train --data data/paired --epochs 100 --resume --output checkpoints
```

Training can be memory-intensive. Reduce `--batch-size` or `--patch-size` if you run out of GPU/CPU memory.

## Evaluation

To compare enhanced images with aligned bright references:

```bash
python scripts/evaluate_pairs.py --predictions outputs/enhanced --references data/paired/val/bright --output reports/metrics.csv
```

The output includes image-level PSNR, SSIM, and MAE. Metrics are meaningful only when prediction and reference images are correctly paired and spatially aligned.

## Reported results

The included `reports/experiment_summary.md` transcribes headline figures from the supplied paper, including a reported best validation PSNR of 20.63 dB at epoch 74 and a reported CLAHE composite score of 0.874. These are **paper-reported figures, not independently reproduced results from this repository**. The original scoring implementation and exact checkpoint have not been independently verified here, so do not present the numbers as a fresh benchmark.

## Tests

After installing dependencies:

```bash
pytest
```

The tests check basic model output shape/range and classical-method output validity. Passing unit tests does not establish that the supplied checkpoint loads correctly or that the reported research results are reproducible.

## Notes and limitations

- The checkpoint must match the model architecture and parameter naming used to train it. If loading fails, verify the checkpoint format and model configuration rather than simply renaming the file.
- The repository does not include the training images or the checkpoint weights.
- Performance depends on the dataset, preprocessing, hardware, and evaluation protocol.
- Please cite the original research or dataset where appropriate and follow the dataset provider's license.

## License

This repository is distributed under the MIT License. Check the license terms of any dataset, pretrained weights, or third-party resources separately before redistribution.
