from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from watermarklab.vendor.mahto2022.watermarking import (
    WatermarkSideInfo,
    embed_full,
    extract_full,
)


@dataclass
class Mahto2022Key:
    side_info: WatermarkSideInfo
    mac: str
    aadhar_dummy: str
    alpha: float
    key: str


class Mahto2022FireflyDual:
    """Mahto & Singh 2022 firefly-optimized dual/multi watermark baseline.

    Paper path:
        R channel: DWT-HH text mark (MAC-like payload)
        G channel: spatial pseudo-magic payload (Aadhaar/hash-like payload)
        B channel: encrypted image watermark in a contourlet/T-SVD-like transform

    The paper does not publish full code for contourlet, SIE encryption, magic-cube
    construction, or all firefly settings.  The vendored implementation follows the
    described channel roles and extraction equations with deterministic substitutes for
    missing implementation-level details.
    """

    name = "Mahto2022_Firefly_Dual"

    def __init__(
        self,
        alpha: float = 0.05,
        mac: str = "AA:BB:CC:DD:EE:FF",
        aadhar_dummy: str = "000000000000",
        key: str = "mahto2022-demo-key",
    ):
        self.alpha = float(alpha)
        self.mac = str(mac)
        self.aadhar_dummy = str(aadhar_dummy)
        self.key = str(key)

    def embed(self, host_rgb: np.ndarray, watermark_binary: np.ndarray):
        wm = np.asarray(watermark_binary, dtype=np.uint8)
        watermarked, side = embed_full(
            host_rgb,
            wm,
            mac=self.mac,
            aadhar_dummy=self.aadhar_dummy,
            alpha=self.alpha,
            key=self.key,
        )
        return watermarked.astype(np.uint8), Mahto2022Key(side, self.mac, self.aadhar_dummy, self.alpha, self.key)

    def extract(self, possibly_attacked_rgb: np.ndarray, key: Mahto2022Key, host_rgb: np.ndarray | None = None):
        result = extract_full(possibly_attacked_rgb, key.side_info, key=key.key)
        wm = np.asarray(result["watermark"], dtype=np.uint8)
        return np.where(wm >= 127, 255, 0).astype(np.uint8)

    def extract_payloads(self, possibly_attacked_rgb: np.ndarray, key: Mahto2022Key):
        return extract_full(possibly_attacked_rgb, key.side_info, key=key.key)


__all__ = ["Mahto2022FireflyDual", "Mahto2022Key"]
