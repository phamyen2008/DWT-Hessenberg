from __future__ import annotations

import math
import numpy as np


def _as_float(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def psnr(original: np.ndarray, test: np.ndarray, data_range: float = 255.0) -> float:
    """Peak signal-to-noise ratio in dB."""
    a = _as_float(original)
    b = _as_float(test)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10((data_range ** 2) / mse)


def nc(original: np.ndarray, recovered: np.ndarray, eps: float = 1e-12) -> float:
    """Uncentered normalized correlation, widely used in watermarking papers."""
    a = _as_float(original).ravel()
    b = _as_float(recovered).ravel()
    denom = math.sqrt(float(np.sum(a * a) * np.sum(b * b)))
    if denom < eps:
        return 0.0
    return float(np.sum(a * b) / denom)


def ncc(original: np.ndarray, recovered: np.ndarray, eps: float = 1e-12) -> float:
    """Centered normalized cross-correlation."""
    a = _as_float(original).ravel()
    b = _as_float(recovered).ravel()
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = math.sqrt(float(np.sum(a * a) * np.sum(b * b)))
    if denom < eps:
        return 0.0
    return float(np.sum(a * b) / denom)


def ber(original_bits: np.ndarray, recovered_bits: np.ndarray, percent: bool = True) -> float:
    """Bit error rate. By default returns percentage, matching paper tables."""
    a = np.asarray(original_bits).astype(np.uint8).ravel()
    b = np.asarray(recovered_bits).astype(np.uint8).ravel()
    if a.shape != b.shape:
        raise ValueError(f"BER requires equal lengths, got {a.shape} vs {b.shape}")
    value = float(np.mean(a != b))
    return 100.0 * value if percent else value


def image_to_bits(img: np.ndarray, threshold: float = 127.5) -> np.ndarray:
    return (_as_float(img) > threshold).astype(np.uint8)


def npcr(a: np.ndarray, b: np.ndarray) -> float:
    """Number of pixel change rate, returned as ratio in [0, 1]."""
    x = np.asarray(a)
    y = np.asarray(b)
    if x.shape != y.shape:
        raise ValueError("NPCR requires equal shapes")
    return float(np.mean(x != y))


def uaci(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    """Unified averaged changed intensity, returned as ratio."""
    x = _as_float(a)
    y = _as_float(b)
    if x.shape != y.shape:
        raise ValueError("UACI requires equal shapes")
    return float(np.mean(np.abs(x - y)) / data_range)


def summarize_metrics(original_cover: np.ndarray, watermarked_cover: np.ndarray,
                      original_wm: np.ndarray, extracted_wm: np.ndarray) -> dict:
    orig_bits = image_to_bits(original_wm)
    ext_bits = image_to_bits(extracted_wm)
    return {
        "psnr": psnr(original_cover, watermarked_cover),
        "nc": nc(original_wm, extracted_wm),
        "ncc": ncc(original_wm, extracted_wm),
        "ber_percent": ber(orig_bits, ext_bits),
    }
