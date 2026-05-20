from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .encryption import encrypt_image, decrypt_image
from .metrics import image_to_bits
from .transforms import (
    dwt2_haar,
    idwt2_haar,
    contourlet_like_tensor,
    inverse_contourlet_like_tensor,
    resize_array_pil,
    tensor_svd_slices,
    reconstruct_tensor_from_svd,
)


def clip_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def string_to_bits(s: str) -> np.ndarray:
    data = s.encode("utf-8")
    bits = []
    for byte in data:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return np.asarray(bits, dtype=np.uint8)


def bits_to_string(bits: np.ndarray) -> str:
    arr = np.asarray(bits, dtype=np.uint8).ravel()
    n = (arr.size // 8) * 8
    arr = arr[:n]
    byts = []
    for i in range(0, n, 8):
        value = 0
        for bit in arr[i:i+8]:
            value = (value << 1) | int(bit)
        byts.append(value)
    return bytes(byts).decode("utf-8", errors="replace")


def int_to_bits(n: int, width: int) -> np.ndarray:
    return np.asarray([(n >> shift) & 1 for shift in range(width - 1, -1, -1)], dtype=np.uint8)


def bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for bit in np.asarray(bits, dtype=np.uint8).ravel():
        value = (value << 1) | int(bit)
    return value


def magic_order(shape: tuple[int, int], seed: int = 2022) -> np.ndarray:
    """Deterministic pseudo-magic spatial permutation.

    The paper references a magic-cube embedding rule but does not specify its full
    construction. This order gives a repeatable scattered spatial embedding path.
    """
    h, w = shape
    idx = np.arange(h * w)
    # Mix row/column structure before RNG permutation: this mimics magic-cube scattering.
    rows = idx // w
    cols = idx % w
    magic_score = (rows * 1315423911 + cols * 2654435761 + rows * cols * 97 + seed) & 0xFFFFFFFF
    order = np.argsort(magic_score, kind="mergesort")
    return order.astype(np.int64)


def blue_hash(blue_channel: np.ndarray, n_hex: int = 16) -> str:
    return hashlib.sha256(np.asarray(blue_channel, dtype=np.uint8).tobytes()).hexdigest()[:n_hex]


@dataclass
class WatermarkSideInfo:
    red: dict[str, Any]
    green: dict[str, Any]
    blue: dict[str, Any]
    alpha: float
    note: str = "Semi-blind side information required by the paper extraction equation."

    def to_jsonable(self) -> dict[str, Any]:
        def convert(obj):
            if isinstance(obj, np.ndarray):
                arr = np.asarray(obj)
                return {
                    "type": "ndarray",
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "min": float(np.min(arr)) if arr.size else None,
                    "max": float(np.max(arr)) if arr.size else None,
                }
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            if isinstance(obj, tuple):
                return list(obj)
            return obj
        return convert({"red": self.red, "green": self.green, "blue": self.blue, "alpha": self.alpha, "note": self.note})


def embed_red_dwt_text(red: np.ndarray, mac: str, strength: float = 6.0) -> tuple[np.ndarray, dict]:
    ll, hl, lh, hh = dwt2_haar(red)
    bits = string_to_bits(mac)
    if bits.size > hh.size:
        raise ValueError("MAC payload too long for red HH band")
    flat = hh.ravel().copy()
    signs = np.where(bits == 1, 1.0, -1.0)
    flat[:bits.size] = flat[:bits.size] + strength * signs
    hh2 = flat.reshape(hh.shape)
    marked = idwt2_haar(ll, hl, lh, hh2)
    return marked, {"mac_len_bits": int(bits.size), "hh_original_prefix": hh.ravel()[:bits.size].copy(), "strength": strength}


def extract_red_dwt_text(marked_red: np.ndarray, side: dict) -> str:
    _, _, _, hh = dwt2_haar(marked_red)
    n = int(side["mac_len_bits"])
    orig = np.asarray(side["hh_original_prefix"], dtype=np.float64)
    diff = hh.ravel()[:n] - orig
    bits = (diff > 0).astype(np.uint8)
    return bits_to_string(bits)


def embed_green_magic_payload(green: np.ndarray, payload: str, seed: int = 2022) -> tuple[np.ndarray, dict]:
    bits = string_to_bits(payload)
    length_bits = int_to_bits(bits.size, 32)
    stream = np.concatenate([length_bits, bits])
    arr = np.asarray(green, dtype=np.uint8).copy()
    order = magic_order(arr.shape, seed)
    if stream.size > order.size:
        raise ValueError("payload too long for green channel LSB embedding")
    flat = arr.ravel()
    positions = order[:stream.size]
    flat[positions] = (flat[positions] & 0xFE) | stream
    return flat.reshape(arr.shape).astype(np.float64), {"seed": seed, "payload_bits": int(bits.size), "total_bits": int(stream.size)}


def extract_green_magic_payload(marked_green: np.ndarray, side: dict) -> str:
    arr = np.asarray(marked_green, dtype=np.uint8)
    order = magic_order(arr.shape, int(side.get("seed", 2022)))
    flat = arr.ravel()
    len_bits = flat[order[:32]] & 1
    payload_len = bits_to_int(len_bits)
    payload_bits = flat[order[32:32 + payload_len]] & 1
    return bits_to_string(payload_bits)


def _svd_to_matrix(u: np.ndarray, s: np.ndarray, vt: np.ndarray) -> np.ndarray:
    return (u * s) @ vt


def embed_blue_contourlet_tsvd(blue: np.ndarray, watermark: np.ndarray, alpha: float,
                               key: str = "mahto2022-demo-key") -> tuple[np.ndarray, dict]:
    """Blue-channel watermark embedding following the paper equation.

    Transform path:
        blue -> contourlet-like coefficient tensor -> per-slice SVD
        watermark -> encrypt -> resize to coefficient size -> SVD
        S_b' = S_b + alpha*S_w
    """
    blue = np.asarray(blue, dtype=np.float64)
    coeff = contourlet_like_tensor(blue)
    coeff_h, coeff_w, k_count = coeff.shape

    wm_orig_shape = tuple(np.asarray(watermark).shape)
    wm_coeff = resize_array_pil(watermark, (coeff_h, coeff_w))
    enc_wm, enc_meta = encrypt_image(clip_uint8(wm_coeff), key=key)
    enc_wm = np.asarray(enc_wm, dtype=np.float64)

    # The paper indicates TSVD on the transformed blue component.  For speed and
    # reproducibility, the executable implementation embeds in the first/low-frequency
    # directional tensor slice, while keeping the full coefficient tensor for inverse
    # reconstruction.
    bu, bs, bvt = np.linalg.svd(coeff[:, :, 0], full_matrices=False)
    wu, ws, wvt = np.linalg.svd(enc_wm, full_matrices=False)

    r = min(bs.size, ws.size)
    bs_marked = bs.copy()
    bs_marked[:r] = bs_marked[:r] + alpha * ws[:r]
    marked_coeff = coeff.copy()
    marked_coeff[:, :, 0] = _svd_to_matrix(bu, bs_marked, bvt)
    marked_blue = inverse_contourlet_like_tensor(marked_coeff)
    side = {
        "alpha": float(alpha),
        "weights": np.asarray([1.0], dtype=np.float64),
        "original_spectra": [bs.copy()],
        "wm_u": wu,
        "wm_vt": wvt,
        "wm_shape_original": wm_orig_shape,
        "wm_shape_coeff": (coeff_h, coeff_w),
        "key": key,
        "enc_meta": enc_meta,
        "blue_lowfreq_u": bu,
        "blue_lowfreq_vt": bvt,
        "wm_singular_values": ws,
    }
    return marked_blue, side


def extract_blue_contourlet_tsvd(marked_blue: np.ndarray, side: dict, key: str | None = None) -> np.ndarray:
    key = side["key"] if key is None else key
    alpha = float(side["alpha"])
    weights = np.asarray(side["weights"], dtype=np.float64)
    original_spectra = side["original_spectra"]
    wm_u = np.asarray(side["wm_u"], dtype=np.float64)
    wm_vt = np.asarray(side["wm_vt"], dtype=np.float64)

    coeff = contourlet_like_tensor(marked_blue)
    _, sm, _ = np.linalg.svd(coeff[:, :, 0], full_matrices=False)
    sb = np.asarray(original_spectra[0], dtype=np.float64)
    r = min(sm.size, sb.size, wm_u.shape[1], wm_vt.shape[0])
    # Paper extraction equation: S_w' = (S_b' - S_b) / alpha.
    s_est = (sm[:r] - sb[:r]) / alpha
    enc_est = _svd_to_matrix(wm_u[:, :r], s_est, wm_vt[:r, :])
    enc_est_u8 = clip_uint8(enc_est)
    wm_coeff = decrypt_image(enc_est_u8, key=key)
    wm_shape_original = tuple(side["wm_shape_original"])
    return resize_array_pil(wm_coeff, wm_shape_original)


def embed_full(cover_rgb: np.ndarray, watermark_gray: np.ndarray, mac: str, aadhar_dummy: str,
               alpha: float = 0.0954, key: str = "mahto2022-demo-key") -> tuple[np.ndarray, WatermarkSideInfo]:
    cover = np.asarray(cover_rgb, dtype=np.uint8)
    if cover.ndim != 3 or cover.shape[2] != 3:
        raise ValueError("cover_rgb must be HxWx3")
    red = cover[:, :, 0].astype(np.float64)
    green = cover[:, :, 1].astype(np.float64)
    blue = cover[:, :, 2].astype(np.float64)

    marked_red, red_side = embed_red_dwt_text(red, mac)
    payload = f"AADHAR_DUMMY={aadhar_dummy};BLUE_HASH={blue_hash(blue)}"
    marked_green, green_side = embed_green_magic_payload(green, payload)
    marked_blue, blue_side = embed_blue_contourlet_tsvd(blue, watermark_gray, alpha=alpha, key=key)

    marked = np.stack([clip_uint8(marked_red), clip_uint8(marked_green), clip_uint8(marked_blue)], axis=2)
    side = WatermarkSideInfo(red=red_side, green=green_side, blue=blue_side, alpha=alpha)
    return marked.astype(np.uint8), side


def extract_full(marked_rgb: np.ndarray, side: WatermarkSideInfo | dict, key: str | None = None) -> dict[str, Any]:
    if isinstance(side, WatermarkSideInfo):
        red_side, green_side, blue_side = side.red, side.green, side.blue
    else:
        red_side, green_side, blue_side = side["red"], side["green"], side["blue"]
    marked = np.asarray(marked_rgb, dtype=np.uint8)
    mac = extract_red_dwt_text(marked[:, :, 0].astype(np.float64), red_side)
    payload = extract_green_magic_payload(marked[:, :, 1], green_side)
    watermark = extract_blue_contourlet_tsvd(marked[:, :, 2].astype(np.float64), blue_side, key=key)
    return {"mac": mac, "payload": payload, "watermark": watermark}


def make_demo_cover(size: int = 512) -> np.ndarray:
    """Generate an ad hoc RGB cover image with smooth colour regions.

    It is not Lena, but intentionally has a face/hat-like structure and smooth
    natural-image gradients similar to Fig. 2 in the paper.
    """
    h = w = size
    y, x = np.mgrid[0:h, 0:w]
    r = 120 + 70 * (x / w) + 25 * np.sin(2 * np.pi * y / h)
    g = 90 + 70 * (y / h) + 20 * np.cos(2 * np.pi * x / w)
    b = 130 + 50 * ((x + y) / (2 * w))
    img = np.stack([r, g, b], axis=2)
    pil = Image.fromarray(clip_uint8(img), "RGB")
    draw = ImageDraw.Draw(pil, "RGBA")
    # face
    draw.ellipse((int(0.34*w), int(0.25*h), int(0.72*w), int(0.78*h)), fill=(230, 170, 145, 220))
    # hat brim/body
    draw.polygon([(int(0.12*w), int(0.25*h)), (int(0.82*w), int(0.08*h)), (int(0.93*w), int(0.22*h)), (int(0.20*w), int(0.38*h))], fill=(180, 105, 120, 210))
    draw.polygon([(int(0.48*w), int(0.02*h)), (int(0.82*w), int(0.08*h)), (int(0.62*w), int(0.26*h))], fill=(210, 150, 120, 200))
    # hair/shadow
    draw.pieslice((int(0.25*w), int(0.20*h), int(0.70*w), int(0.82*h)), 95, 260, fill=(70, 55, 75, 180))
    # eyes/mouth
    draw.ellipse((int(0.49*w), int(0.48*h), int(0.53*w), int(0.52*h)), fill=(30, 30, 45, 230))
    draw.ellipse((int(0.61*w), int(0.47*h), int(0.65*w), int(0.51*h)), fill=(30, 30, 45, 230))
    draw.arc((int(0.53*w), int(0.60*h), int(0.66*w), int(0.68*h)), 10, 170, fill=(120, 40, 55, 220), width=max(1, size//150))
    pil = pil.filter(ImageFilter.GaussianBlur(radius=0.4))
    return np.asarray(pil, dtype=np.uint8)


def make_demo_watermark(size: int = 128) -> np.ndarray:
    """Generate a Cameraman-like grayscale watermark."""
    h = w = size
    y, x = np.mgrid[0:h, 0:w]
    base = 180 + 45 * np.sin(2 * np.pi * x / w) * np.cos(2 * np.pi * y / h)
    pil = Image.fromarray(clip_uint8(base), "L")
    draw = ImageDraw.Draw(pil)
    # ground/sky
    draw.rectangle((0, int(0.72*h), w, h), fill=120)
    # person
    draw.ellipse((int(0.22*w), int(0.18*h), int(0.35*w), int(0.31*h)), fill=30)
    draw.rectangle((int(0.26*w), int(0.30*h), int(0.36*w), int(0.58*h)), fill=35)
    draw.line((int(0.28*w), int(0.56*h), int(0.20*w), int(0.78*h)), fill=25, width=max(1, size//32))
    draw.line((int(0.34*w), int(0.56*h), int(0.44*w), int(0.78*h)), fill=25, width=max(1, size//32))
    # camera and tripod
    draw.rectangle((int(0.38*w), int(0.32*h), int(0.57*w), int(0.42*h)), fill=20)
    draw.ellipse((int(0.51*w), int(0.33*h), int(0.60*w), int(0.42*h)), fill=65)
    draw.line((int(0.48*w), int(0.43*h), int(0.42*w), int(0.78*h)), fill=20, width=max(1, size//48))
    draw.line((int(0.48*w), int(0.43*h), int(0.56*w), int(0.78*h)), fill=20, width=max(1, size//48))
    draw.line((int(0.48*w), int(0.43*h), int(0.49*w), int(0.78*h)), fill=20, width=max(1, size//48))
    # distant horizon
    draw.line((0, int(0.70*h), w, int(0.68*h)), fill=90, width=1)
    return np.asarray(pil.filter(ImageFilter.GaussianBlur(radius=0.2)), dtype=np.uint8)


def save_image(path: str, arr: np.ndarray) -> None:
    Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(path)
