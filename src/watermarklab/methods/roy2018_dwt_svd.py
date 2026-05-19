from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from watermarklab.common.color import rgb_to_ycbcr, ycbcr_to_rgb
from watermarklab.common.dwt import multilevel_dwt_ll, multilevel_idwt_ll

@dataclass
class RoyKey:
    original_s: list[np.ndarray]
    uw: list[np.ndarray]
    vwt: list[np.ndarray]
    alpha: float


def _split_blocks(arr: np.ndarray, block_size: int):
    h, w = arr.shape[:2]
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            block = arr[y:y+block_size, x:x+block_size]
            if block.shape[:2] == (block_size, block_size):
                yield y, x, block


class Roy2018DWTSVD:
    name = "Roy2018_DWT_SVD"

    def __init__(self, alpha: float = 0.02):
        self.alpha = float(alpha)

    def embed(self, host_rgb: np.ndarray, watermark_binary: np.ndarray):
        y, cb, cr = rgb_to_ycbcr(host_rgb)
        y2 = y.copy()
        original_s: list[np.ndarray] = []
        uw_list: list[np.ndarray] = []
        vwt_list: list[np.ndarray] = []
        wm_blocks = [b.astype(np.float64) for _, _, b in _split_blocks(watermark_binary, 4)]
        if len(wm_blocks) != 256:
            raise ValueError("Roy2018 expects a 64x64 watermark split into 256 blocks")
        idx = 0
        for by, bx, block in _split_blocks(y, 32):
            ll, details = multilevel_dwt_ll(block, levels=3, mode="average")
            u, s_vals, vt = np.linalg.svd(ll, full_matrices=False)
            s_mat = np.diag(s_vals)
            sw_mix = s_mat + self.alpha * wm_blocks[idx]
            uw, sw_vals, vwt = np.linalg.svd(sw_mix, full_matrices=False)
            ll_marked = u @ np.diag(sw_vals) @ vt
            y2[by:by+32, bx:bx+32] = multilevel_idwt_ll(ll_marked, details, mode="average")
            original_s.append(s_mat)
            uw_list.append(uw)
            vwt_list.append(vwt)
            idx += 1
        return ycbcr_to_rgb(y2, cb, cr), RoyKey(original_s, uw_list, vwt_list, self.alpha)

    def extract(self, possibly_attacked_rgb: np.ndarray, key: RoyKey, host_rgb: np.ndarray | None = None):
        y, _, _ = rgb_to_ycbcr(possibly_attacked_rgb)
        recovered = np.zeros((64, 64), dtype=np.float64)
        idx = 0
        for by, bx, block in _split_blocks(y, 32):
            ll, _ = multilevel_dwt_ll(block, levels=3, mode="average")
            _, s_vals, _ = np.linalg.svd(ll, full_matrices=False)
            s_mat = np.diag(s_vals)
            mixed = key.uw[idx] @ s_mat @ key.vwt[idx]
            rec = (mixed - key.original_s[idx]) / key.alpha
            wy = (idx // 16) * 4
            wx = (idx % 16) * 4
            recovered[wy:wy+4, wx:wx+4] = rec
            idx += 1
        return np.where(np.clip(recovered, 0, 255) >= 127, 255, 0).astype(np.uint8)
