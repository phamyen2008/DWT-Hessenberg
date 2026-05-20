from __future__ import annotations

import numpy as np


def dwt2_haar(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Single-level 2D Haar DWT with orthonormal scaling.

    Returns LL, HL, LH, HH where HL responds to vertical detail and LH to horizontal detail.
    Input dimensions must be even.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("dwt2_haar expects a 2D array")
    h, w = arr.shape
    if h % 2 or w % 2:
        raise ValueError("dwt2_haar requires even height and width")
    a = arr[0::2, 0::2]
    b = arr[0::2, 1::2]
    c = arr[1::2, 0::2]
    d = arr[1::2, 1::2]
    ll = (a + b + c + d) / 2.0
    hl = (a - b + c - d) / 2.0
    lh = (a + b - c - d) / 2.0
    hh = (a - b - c + d) / 2.0
    return ll, hl, lh, hh


def idwt2_haar(ll: np.ndarray, hl: np.ndarray, lh: np.ndarray, hh: np.ndarray) -> np.ndarray:
    """Inverse of `dwt2_haar`."""
    ll = np.asarray(ll, dtype=np.float64)
    hl = np.asarray(hl, dtype=np.float64)
    lh = np.asarray(lh, dtype=np.float64)
    hh = np.asarray(hh, dtype=np.float64)
    if not (ll.shape == hl.shape == lh.shape == hh.shape):
        raise ValueError("all DWT subbands must have the same shape")
    a = (ll + hl + lh + hh) / 2.0
    b = (ll - hl + lh - hh) / 2.0
    c = (ll + hl - lh - hh) / 2.0
    d = (ll - hl - lh + hh) / 2.0
    h, w = ll.shape
    out = np.zeros((2 * h, 2 * w), dtype=np.float64)
    out[0::2, 0::2] = a
    out[0::2, 1::2] = b
    out[1::2, 0::2] = c
    out[1::2, 1::2] = d
    return out


def contourlet_like_tensor(x: np.ndarray) -> np.ndarray:
    """Directional coefficient tensor used as a contourlet surrogate.

    The paper uses CT followed by Tensor-SVD but omits the exact CT filter bank.
    This deterministic replacement stacks the four Haar directional subbands as a
    coefficient tensor, preserving the transform-domain/tensor-SVD structure.
    """
    ll, hl, lh, hh = dwt2_haar(x)
    return np.stack([ll, hl, lh, hh], axis=2)


def inverse_contourlet_like_tensor(tensor: np.ndarray) -> np.ndarray:
    arr = np.asarray(tensor, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] != 4:
        raise ValueError("expected tensor with shape (H/2, W/2, 4)")
    return idwt2_haar(arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3])


def tensor_svd_slices(tensor: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per-slice SVD for a 3D coefficient tensor.

    Returns a list of (U, S, Vt), one for each tensor slice.
    """
    arr = np.asarray(tensor, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError("tensor_svd_slices expects a 3D tensor")
    factors = []
    for k in range(arr.shape[2]):
        u, s, vt = np.linalg.svd(arr[:, :, k], full_matrices=False)
        factors.append((u, s, vt))
    return factors


def reconstruct_tensor_from_svd(factors: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    slices = []
    for u, s, vt in factors:
        slices.append((u * s) @ vt)
    return np.stack(slices, axis=2)


def resize_array_pil(x: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    from PIL import Image
    img = Image.fromarray(np.clip(x, 0, 255).astype(np.uint8))
    img = img.resize((shape[1], shape[0]), Image.Resampling.BICUBIC)
    return np.asarray(img, dtype=np.float64)
