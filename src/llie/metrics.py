"""Reference-based image-quality metrics for paired evaluation."""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def image_metrics(prediction, reference):
    """Inputs are same-size RGB uint8 arrays or PIL images."""
    pred = np.asarray(prediction.convert("RGB") if hasattr(prediction, "convert") else prediction)
    ref = np.asarray(reference.convert("RGB") if hasattr(reference, "convert") else reference)
    if pred.shape != ref.shape: raise ValueError(f"Image shapes differ: {pred.shape} != {ref.shape}")
    pred = pred.astype(np.uint8); ref = ref.astype(np.uint8)
    return {
        "psnr_db": float(peak_signal_noise_ratio(ref, pred, data_range=255)),
        "ssim": float(structural_similarity(ref, pred, channel_axis=2, data_range=255)),
        "mae": float(np.abs(pred.astype(np.float32) - ref.astype(np.float32)).mean()),
    }
