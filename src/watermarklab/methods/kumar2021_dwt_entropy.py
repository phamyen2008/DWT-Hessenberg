from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from watermarklab.common.color import rgb_to_ycbcr, ycbcr_to_rgb
from watermarklab.common.iwt import iwt2, iiwt2
from watermarklab.common.chaos import arnold_scramble, arnold_unscramble
from watermarklab.common.embedding_math import alpha_blend_cover_weight, extract_from_alpha_blend_cover_weight
from watermarklab.common.entropy import (
    visual_entropy,
    edge_entropy,
    block_entropy_score,
    select_max_entropy_score_block,
)

@dataclass
class KumarKey:
    alpha: float
    block_index: tuple[int, int]
    block_size: int
    watermark_ll: np.ndarray
    watermark_lh: np.ndarray
    watermark_hl: np.ndarray
    arnold_iterations: int


def shannon_entropy(block: np.ndarray, bins: int = 256) -> float:
    """Backward-compatible alias for information/visual entropy."""
    return visual_entropy(block, bins=bins)


def select_max_entropy_block(hh: np.ndarray, block_size: int = 32):
    """Select the block maximizing B_E = E_v - E_e as described by Kumar et al."""
    return select_max_entropy_score_block(hh, block_size=block_size)


class Kumar2021DWTEntropy:
    """Kumar & Singh 2021 LWT/entropy/ACM alpha-blending baseline.

    The class name is kept for backward compatibility with earlier project code,
    but the implementation uses the LWT/IWT module, visual+edge entropy block
    selection, Arnold Cat Map security, and alpha blending in the selected HH
    block of the Y component.
    """

    name = "Kumar2021_LWT_Entropy"

    def __init__(self, alpha: float = 0.99, block_size: int = 32, arnold_iterations: int = 50):
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = float(alpha)
        self.block_size = int(block_size)
        self.arnold_iterations = int(arnold_iterations)

    def embed(self, host_rgb: np.ndarray, watermark_binary: np.ndarray):
        y, cb, cr = rgb_to_ycbcr(host_rgb)
        ll, lh, hl, hh = iwt2(y)

        wm = np.asarray(watermark_binary, dtype=np.uint8)
        if wm.shape != (64, 64):
            raise ValueError("Kumar2021 common mode expects a 64x64 watermark")
        wm_scrambled = arnold_scramble(wm, iterations=self.arnold_iterations).astype(np.float64)
        w_ll, w_lh, w_hl, w_hh = iwt2(wm_scrambled)

        (bi, bj), _ = select_max_entropy_block(hh, self.block_size)
        r = bi * self.block_size
        c = bj * self.block_size
        hh2 = hh.copy()
        # Alpha blending in LWT-HH: marked = alpha*cover + (1-alpha)*encrypted watermark.
        hh2[r:r+self.block_size, c:c+self.block_size] = alpha_blend_cover_weight(
            hh[r:r+self.block_size, c:c+self.block_size], w_hh, self.alpha
        )
        y2 = iiwt2(ll, lh, hl, hh2)
        watermarked = ycbcr_to_rgb(y2, cb, cr)
        key = KumarKey(self.alpha, (bi, bj), self.block_size, w_ll, w_lh, w_hl, self.arnold_iterations)
        return watermarked, key

    def extract(self, possibly_attacked_rgb: np.ndarray, key: KumarKey, host_rgb: np.ndarray | None = None):
        if host_rgb is None:
            raise ValueError("Kumar2021 extraction is non-blind and requires host_rgb")
        bi, bj = key.block_index
        r = bi * key.block_size
        c = bj * key.block_size
        y_wm, _, _ = rgb_to_ycbcr(possibly_attacked_rgb)
        y_host, _, _ = rgb_to_ycbcr(host_rgb)
        _, _, _, hh_wm = iwt2(y_wm)
        _, _, _, hh_host = iwt2(y_host)
        extracted_hh = extract_from_alpha_blend_cover_weight(
            hh_wm[r:r+key.block_size, c:c+key.block_size],
            hh_host[r:r+key.block_size, c:c+key.block_size],
            key.alpha,
        )
        scrambled_rec = iiwt2(key.watermark_ll, key.watermark_lh, key.watermark_hl, extracted_hh)
        recovered = arnold_unscramble(np.clip(np.rint(scrambled_rec), 0, 255).astype(np.uint8), iterations=key.arnold_iterations)
        return np.where(recovered >= 127, 255, 0).astype(np.uint8)


__all__ = [
    "Kumar2021DWTEntropy",
    "KumarKey",
    "shannon_entropy",
    "visual_entropy",
    "edge_entropy",
    "block_entropy_score",
    "select_max_entropy_block",
]
