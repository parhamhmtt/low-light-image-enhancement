"""Classical low-light enhancement baselines implemented with OpenCV/NumPy."""
import cv2
import numpy as np


def _uint8(image):
    if isinstance(image, str): image = cv2.imread(image, cv2.IMREAD_COLOR)
    if image is None: raise ValueError("Could not read image")
    if image.dtype != np.uint8: image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def histogram_equalization(image):
    image = _uint8(image); ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    image = _uint8(image); lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def gamma_correction(image, gamma=None):
    image = _uint8(image)
    if gamma is None:
        luminance = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean() / 255.0
        gamma = np.clip(np.log(max(luminance, 1e-4) / 0.5) / np.log(0.5), 0.4, 6.0)
    inv = 1.0 / float(gamma)
    table = np.clip((np.arange(256) / 255.0) ** inv * 255, 0, 255).astype(np.uint8)
    return cv2.LUT(image, table)


def single_scale_retinex(image, sigma=30):
    image = _uint8(image).astype(np.float32) + 1.0
    blur = cv2.GaussianBlur(image, (0, 0), sigma) + 1.0
    result = np.log(image) - np.log(blur)
    out = np.empty_like(result)
    for c in range(3):
        channel = result[:, :, c]; lo, hi = np.percentile(channel, (1, 99))
        out[:, :, c] = np.clip((channel - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
    return out.astype(np.uint8)


def multi_scale_retinex(image, sigmas=(15, 80, 250)):
    image = _uint8(image).astype(np.float32) + 1.0
    log_image = np.log(image); result = np.zeros_like(image)
    for sigma in sigmas:
        blur = cv2.GaussianBlur(image, (0, 0), sigma) + 1.0
        result += log_image - np.log(blur)
    result /= len(sigmas); out = np.empty_like(result)
    for c in range(3):
        lo, hi = np.percentile(result[:, :, c], (1, 99))
        out[:, :, c] = np.clip((result[:, :, c] - lo) / max(hi - lo, 1e-6) * 255, 0, 255)
    return out.astype(np.uint8)


def msrcr(image, sigmas=(15, 80, 250), gain=192, offset=-30):
    image = _uint8(image).astype(np.float32) + 1.0
    msr = np.zeros_like(image); log_sum = np.log(np.sum(image, axis=2, keepdims=True))
    for sigma in sigmas:
        blur = cv2.GaussianBlur(image, (0, 0), sigma) + 1.0
        msr += np.log(image) - np.log(blur)
    msr /= len(sigmas)
    color_restore = np.log(125.0 * image) - log_sum
    result = gain * (msr * color_restore + offset / max(gain, 1))
    lo, hi = np.percentile(result, (1, 99))
    return np.clip((result - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)


def exposure_fusion(image):
    image = _uint8(image)
    candidates = [gamma_correction(image, g) for g in (0.5, 1.0, 1.5)]
    candidates.append(clahe(image, clip_limit=3.0))
    weights = []
    for candidate in candidates:
        f = candidate.astype(np.float32) / 255.0
        gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        contrast = np.abs(cv2.Laplacian(gray, cv2.CV_32F)) + 1e-3
        saturation = np.std(f, axis=2) + 1e-3
        well_exposed = np.exp(-0.5 * ((gray - 0.5) / 0.2) ** 2) + 1e-3
        weights.append(contrast * saturation * well_exposed)
    weights = np.stack(weights, axis=0)
    weights /= np.sum(weights, axis=0, keepdims=True) + 1e-8
    fused = sum(c.astype(np.float32) * weights[i, :, :, None] for i, c in enumerate(candidates))
    return np.clip(fused, 0, 255).astype(np.uint8)


METHODS = {
    "he": histogram_equalization, "clahe": clahe, "gamma": gamma_correction,
    "ssr": single_scale_retinex, "msr": multi_scale_retinex, "msrcr": msrcr,
    "fusion": exposure_fusion,
}
