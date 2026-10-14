# Low-Light Image Enhancement

**Classical image-processing baselines and a trained residual U-Net for improving visibility in dark photographs.**

This repository organizes the code and research from *Low-Light Image Enhancement: A Comparative Study of Classical Algorithms and a Trained Residual U-Net* into a reproducible, reusable project. It includes classical enhancement methods, a PyTorch model, training and inference commands, paired-image evaluation, and the accompanying research paper.

> **Model weights and project resources:** The trained `best.pth` checkpoint and other supplied files are hosted on [Google Drive](https://drive.google.com/drive/folders/1I-YwALN_dvdBpt0yApZlVd_Vhtkiy2s0?usp=sharing). Download the checkpoint and place it at `checkpoints/best.pth`. The checkpoint is not committed to Git because model files can be large.

## Contents

- **Residual U-Net:** encoder-decoder architecture with skip connections, residual blocks, InstanceNorm, LeakyReLU, and sigmoid RGB output.
- **Classical baselines:** Histogram Equalization (HE), CLAHE, Gamma Correction, Single-Scale Retinex (SSR), Multi-Scale Retinex (MSR), MSR with Color Restoration (MSRCR), and Exposure Fusion.
- **Training:** paired dark/bright dataset loader, synchronized augmentation, L1 objective, AdamW, cosine learning-rate schedule, gradient clipping, validation loss and PSNR, best/last checkpoints, and JSON history.
- **Inference:** process one image or a whole directory; automatically uses CUDA, Apple MPS, or CPU when available.
- **Evaluation:** PSNR, SSIM, and MAE for enhanced images with paired references.
- **Research:** the paper is included in [`docs/`](docs/).

## Repository layout

```text
low-light-image-enhancement/
├── src/llie/
│   ├── model.py          # Residual U-Net
│   ├── data.py           # Paired image dataset
│   ├── train.py          # Training loop and checkpointing
│   ├── inference.py      # Model loading and inference utilities
│   ├── classical.py      # Classical enhancement methods
│   └── metrics.py        # PSNR, SSIM, MAE
├── scripts/
│   ├── enhance.py
│   ├── enhance_classical.py
│   └── evaluate_pairs.py
├── tests/                # Unit tests
├── docs/                 # Research paper (PDF and DOCX)
├── data/paired/          # Add dataset locally; excluded from Git
├── checkpoints/          # Add best.pth locally; excluded from Git
├── outputs/               # Generated results; excluded from Git
├── requirements.txt
└── pyproject.toml
```

## Installation

Python 3.10 or newer is recommended. Create and activate a virtual environment first.

**Windows PowerShell**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

For GPU training, install the PyTorch and torchvision builds compatible with your CUDA setup using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

## Quick start: enhance a dark image

1. Download `best.pth` from the [project Google Drive folder](https://drive.google.com/drive/folders/1I-YwALN_dvdBpt0yApZlVd_Vhtkiy2s0?usp=sharing).
2. Create `checkpoints/` in the repository root and put the file there.
3. Run:

```bash
python scripts/enhance.py --input path/to/dark_photo.jpg --checkpoint checkpoints/best.pth --output outputs/enhanced
```

Enhance every supported image in a folder:

```bash
python scripts/enhance.py --input path/to/dark_images/ --output outputs/enhanced/
```

Choose the compute device explicitly if needed:

```bash
python scripts/enhance.py --input input.jpg --device cpu
# or --device cuda / --device mps
```

The model writes PNG files named `enhanced_<original-name>.png`. It pads dimensions to a multiple of 16 for the U-Net and crops the result back to the original size.

**Checkpoint compatibility:** the checkpoint must match this repository's `LowLightUNet` architecture. Training checkpoints store the model configuration. If you have a plain state dictionary rather than a training checkpoint, specify `--base-channels 32` or `--base-channels 64` to match the model used to train it. If loading fails, check the checkpoint format and architecture instead of renaming the file.

## Classical enhancement baselines

```bash
python scripts/enhance_classical.py --input path/to/dark_photo.jpg --method clahe --output outputs/clahe.png
```

Supported `--method` values: `he`, `clahe`, `gamma`, `ssr`, `msr`, `msrcr`, and `fusion`.

Python API example:

```python
import cv2
from llie.classical import clahe, exposure_fusion

image = cv2.imread("dark_photo.jpg")  # OpenCV loads BGR images
cv2.imwrite("clahe.png", clahe(image))
cv2.imwrite("fusion.png", exposure_fusion(image))
```

## Dataset format

The training loader expects paired images with identical filenames and dimensions. Keep the original low-light and normal-light images aligned.

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

Use a dataset you are permitted to access and redistribute. The paper reports experiments on the LOL paired low-light dataset. Dataset files are not included in this repository; follow the dataset provider's terms and prepare the folder structure above.

## Train from scratch or fine-tune

```bash
python -m llie.train --data data/paired --epochs 100 --batch-size 4 --patch-size 256 --base-channels 32 --output checkpoints
```

Resume from `checkpoints/last.pth`:

```bash
python -m llie.train --data data/paired --epochs 100 --resume --output checkpoints
```

Useful options:

| Option | Default | Description |
|---|---:|---|
| `--epochs` | `100` | Number of training epochs |
| `--batch-size` | `4` | Batch size; lower it if memory is limited |
| `--patch-size` | `256` | Random crop size during training |
| `--base-channels` | `32` | U-Net width; must match a checkpoint for inference |
| `--lr` | `0.0002` | Initial learning rate |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, or `mps` |
| `--workers` | `0` | DataLoader workers; start with 0 on Windows |
| `--resume` | off | Resume from the last checkpoint |

Training writes `best.pth`, `last.pth`, and `history.json` to the output directory. The best checkpoint is selected by validation L1 loss. Validation PSNR is logged for monitoring.

## Evaluate results

Place enhanced predictions and their bright reference images in separate directories. Reference names should match prediction names; for files named `enhanced_scene.png`, a reference named `scene.png` is also recognized.

```bash
python scripts/evaluate_pairs.py --predictions outputs/enhanced --references data/paired/test/bright --output reports/metrics.csv
```

The script reports:

- **PSNR (dB):** pixel-level reconstruction fidelity; higher is generally better.
- **SSIM:** structural similarity; higher is generally better.
- **MAE:** mean absolute pixel error; lower is better.

These reference-based metrics require aligned ground-truth images and should be complemented by visual inspection. Do not compare metrics across different datasets or preprocessing settings without noting the differences.

## Reported research results

The supplied paper reports the following results from its experiments:

| Experiment | Reported result | Context |
|---|---:|---|
| Best U-Net validation PSNR | 20.63 dB | Epoch 74 on the reported validation split |
| Best U-Net validation loss | 0.0732 | L1 loss, epoch 74 |
| CLAHE composite score | 0.874 | Proposed multi-dimensional score for classical methods |
| Exposure Fusion composite score | 0.872 | Same reported classical-method comparison |

These values are transcribed from the supplied research paper and are **not a guarantee** that a fresh run will reproduce them. Dataset splits, checkpoint configuration, hardware, random seeds, and implementation details affect results. The paper also notes the small validation set and limited cross-domain evaluation. See [`docs/research_paper.pdf`](docs/research_paper.pdf) for methodology, results, and limitations.

## Reproducibility notes and limitations

- The default loss is L1, matching the final training formulation described in the paper.
- The training pipeline uses deterministic seeds for Python, NumPy, and PyTorch, but full bit-for-bit determinism is not guaranteed across hardware and library versions.
- The model was reported to be trained on paired indoor low-light imagery. Outdoor night scenes, extreme sensor noise, and other domains require separate validation.
- Enhancement can brighten noise or introduce color/contrast artifacts. Always inspect results before using them in safety-critical or scientific settings.
- The classical composite score described in the paper is a research-specific metric and is not presented as a universal image-quality standard.

## Research paper

- [PDF](docs/research_paper.pdf)
- [Editable DOCX](docs/research_paper.docx)

## Citation

If you use the research or code, cite the accompanying paper and replace the placeholder author/year details with the final publication information if it is published:

```text
"Low-Light Image Enhancement: A Comparative Study of Classical Algorithms and a Trained Residual U-Net."
B.Sc. Student Research Project, Faculty of Engineering, University of Guilan, Rasht, Iran.
```

## License

The repository code is released under the MIT License. Dataset licenses and third-party materials remain subject to their original terms.

Run every classical method and generate a visual comparison panel for one image:

```bash
python scripts/compare_classical.py --input path/to/dark_photo.jpg --output-dir outputs/classical_comparison
```

The reported training trajectory and classical-method table are also transcribed in [`reports/experiment_summary.md`](reports/experiment_summary.md).
