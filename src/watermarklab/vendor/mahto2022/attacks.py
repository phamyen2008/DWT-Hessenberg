from __future__ import annotations

from io import BytesIO
import numpy as np
from PIL import Image
from scipy.ndimage import median_filter


def _clip_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def speckle(img: np.ndarray, variance: float = 0.01, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = np.asarray(img, dtype=np.float64)
    noise = rng.normal(0, np.sqrt(variance), arr.shape)
    return _clip_uint8(arr + arr * noise)


def salt_pepper(img: np.ndarray, amount: float = 0.005, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.asarray(img).copy()
    mask = rng.random(out.shape[:2])
    salt = mask < amount / 2
    pepper = (mask >= amount / 2) & (mask < amount)
    if out.ndim == 3:
        out[salt, :] = 255
        out[pepper, :] = 0
    else:
        out[salt] = 255
        out[pepper] = 0
    return out.astype(np.uint8)


def median(img: np.ndarray, size: int = 2) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 3:
        filtered = np.zeros_like(arr)
        for c in range(arr.shape[2]):
            filtered[:, :, c] = median_filter(arr[:, :, c], size=size)
        return filtered.astype(np.uint8)
    return median_filter(arr, size=size).astype(np.uint8)


def hist_equalization(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.uint8)
    out = np.empty_like(arr)
    if arr.ndim == 2:
        channels = [arr]
    else:
        channels = [arr[:, :, i] for i in range(arr.shape[2])]
    eq_channels = []
    for ch in channels:
        hist, _ = np.histogram(ch.ravel(), bins=256, range=(0, 255))
        cdf = hist.cumsum().astype(np.float64)
        nonzero = cdf[cdf > 0]
        if nonzero.size == 0:
            eq = ch.copy()
        else:
            cdf_min = nonzero[0]
            lut = np.rint((cdf - cdf_min) / max(1.0, (cdf[-1] - cdf_min)) * 255.0)
            lut = np.clip(lut, 0, 255).astype(np.uint8)
            eq = lut[ch]
        eq_channels.append(eq)
    if arr.ndim == 2:
        return eq_channels[0]
    return np.stack(eq_channels, axis=2)


def gaussian(img: np.ndarray, variance: float = 0.001, seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = np.asarray(img, dtype=np.float64)
    sigma = np.sqrt(variance) * 255.0
    return _clip_uint8(arr + rng.normal(0, sigma, arr.shape))


def sharpen(img: np.ndarray, amount: float = 0.1) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    arr = np.asarray(img, dtype=np.float64)
    blurred = gaussian_filter(arr, sigma=(1, 1, 0) if arr.ndim == 3 else 1)
    return _clip_uint8(arr + amount * (arr - blurred))


def jpeg_compress(img: np.ndarray, quality: int = 90) -> np.ndarray:
    pil = Image.fromarray(np.asarray(img, dtype=np.uint8))
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert(pil.mode), dtype=np.uint8)


def crop_restore(img: np.ndarray, box: tuple[int, int, int, int] = (20, 20, 400, 480)) -> np.ndarray:
    """Crop a rectangle and paste it back on black canvas of original size."""
    arr = np.asarray(img, dtype=np.uint8)
    x0, y0, x1, y1 = box
    out = np.zeros_like(arr)
    crop = arr[y0:y1, x0:x1].copy()
    out[y0:y1, x0:x1] = crop
    return out


def all_paper_attacks(img: np.ndarray, seed: int = 123) -> dict[str, np.ndarray]:
    return {
        "no_attack": np.asarray(img, dtype=np.uint8),
        "speckle_0.01": speckle(img, 0.01, seed),
        "salt_pepper_0.005": salt_pepper(img, 0.005, seed),
        "median_2x2": median(img, 2),
        "hist_equalization": hist_equalization(img),
        "gaussian_0.001": gaussian(img, 0.001, seed),
        "sharpen_0.1": sharpen(img, 0.1),
        "jpeg_qf90": jpeg_compress(img, 90),
        "crop_20_20_400_480": crop_restore(img, (20, 20, min(400, img.shape[1]), min(480, img.shape[0]))),
    }
