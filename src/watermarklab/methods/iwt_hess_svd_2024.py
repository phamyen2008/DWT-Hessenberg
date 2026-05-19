from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from PIL import Image
from watermarklab.common.iwt import multilevel_iwt, multilevel_iiwt, IWTLevel
from watermarklab.common.chaos import chaotic_encrypt_uint8, chaotic_decrypt_uint8
from watermarklab.common.linalg_utils import hess_decompose, hess_reconstruct, svd_pc

@dataclass
class IWTHessKey:
    alpha: float
    cover_levels: list[IWTLevel]
    q_cover: np.ndarray
    vt_cover: np.ndarray
    pc_cover: np.ndarray
    wm_levels: list[IWTLevel]
    wm_vt: np.ndarray
    wm_shape_large: tuple[int, int]
    wm_shape_out: tuple[int, int]
    chaotic_indices: np.ndarray
    chaotic_mask: np.ndarray


def _resize_nearest(gray: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(np.asarray(gray, dtype=np.uint8), mode="L").resize(size[::-1], Image.Resampling.NEAREST), dtype=np.uint8)


class IWTHessSVD2024:
    name = "IWT_Hess_SVD_2024"

    def __init__(self, alpha: float = 0.015, internal_wm_size: int = 256):
        self.alpha = float(alpha)
        self.internal_wm_size = int(internal_wm_size)

    def embed(self, host_rgb: np.ndarray, watermark_binary: np.ndarray):
        cover = np.asarray(host_rgb, dtype=np.float64)
        # The original paper uses a 256x256 watermark; adapt 64x64 binary input by nearest upsampling.
        wm_large = _resize_nearest(watermark_binary, (self.internal_wm_size, self.internal_wm_size)).astype(np.float64)
        red = cover[:, :, 0]
        cover_ll, cover_levels = multilevel_iwt(red, levels=3)  # 512 -> 64
        q_cover, h_cover = hess_decompose(cover_ll)
        pc_cover, vt_cover = svd_pc(h_cover)

        wm_ll, wm_levels = multilevel_iwt(wm_large, levels=2)   # 256 -> 64
        wm_enc, idx, mask = chaotic_encrypt_uint8(wm_ll)
        pc_wm, wm_vt = svd_pc(wm_enc)

        pc_marked = pc_cover + self.alpha * pc_wm
        h_marked = pc_marked @ vt_cover
        ll_marked = hess_reconstruct(q_cover, h_marked)
        marked_red = multilevel_iiwt(ll_marked, cover_levels)
        out = cover.copy()
        out[:, :, 0] = marked_red
        out = np.clip(np.rint(out), 0, 255).astype(np.uint8)
        key = IWTHessKey(self.alpha, cover_levels, q_cover, vt_cover, pc_cover, wm_levels, wm_vt, wm_large.shape, watermark_binary.shape, idx, mask)
        return out, key

    def extract(self, possibly_attacked_rgb: np.ndarray, key: IWTHessKey, host_rgb: np.ndarray | None = None):
        img = np.asarray(possibly_attacked_rgb, dtype=np.float64)
        red = img[:, :, 0]
        ll_wm, _ = multilevel_iwt(red, levels=3)
        # Non-blind correction: use the original Hessenberg Q and Vt side information
        # stored at embedding time. Recomputing Hessenberg on the watermarked/attacked
        # image can permute the decomposition and severely degrade extraction.
        h_marked_est = key.q_cover.T @ ll_wm @ key.q_cover
        pc_wm_marked = h_marked_est @ key.vt_cover.T
        pc_ext = (pc_wm_marked - key.pc_cover) / key.alpha
        enc_ll = pc_ext @ key.wm_vt
        ll_ext = chaotic_decrypt_uint8(enc_ll, key.chaotic_indices, key.chaotic_mask)
        wm_large = multilevel_iiwt(ll_ext, key.wm_levels)
        wm_out = _resize_nearest(np.clip(np.rint(wm_large), 0, 255).astype(np.uint8), key.wm_shape_out)
        return np.where(wm_out >= 127, 255, 0).astype(np.uint8)
