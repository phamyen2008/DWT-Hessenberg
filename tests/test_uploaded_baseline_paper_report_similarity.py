from pathlib import Path

import numpy as np

from watermarklab.common.io_utils import load_host_rgb, load_watermark_binary
from watermarklab.common.metrics import psnr, ssim, nc, ncc, ber
from watermarklab.methods import build_methods
from watermarklab.methods.gaata2022_dwt_hess_fwa import Gaata2022DWTHessFWA
from watermarklab.methods.mahto2022_firefly_dual import Mahto2022FireflyDual
from watermarklab.methods.kumar2021_dwt_entropy import Kumar2021DWTEntropy
from watermarklab.methods.dwt_hd_svd2025 import DWTHDSVD2025

ROOT = Path(__file__).resolve().parents[1]


def _host_wm():
    return (
        load_host_rgb(ROOT / "data" / "host" / "lenna.bmp"),
        load_watermark_binary(ROOT / "data" / "watermark" / "wm.png"),
    )


def _metric_row(method, host, wm):
    watermarked, key = method.embed(host, wm)
    extracted = method.extract(watermarked, key, host_rgb=host)
    return {
        "method": method.name,
        "watermarked": watermarked,
        "key": key,
        "extracted": extracted,
        "psnr": psnr(host, watermarked),
        "ssim": ssim(host, watermarked),
        "nc": nc(wm, extracted),
        "ncc": ncc(wm, extracted),
        "ber": ber(wm, extracted),
    }


def test_four_uploaded_baselines_are_registered_for_common_benchmark():
    methods = build_methods([
        "kumar2021",
        "gaata2022_dwt_hess_fwa",
        "mahto2022_firefly_dual",
        "dwt_hd_svd_2025",
    ])
    assert list(methods) == [
        "kumar2021",
        "gaata2022_dwt_hess_fwa",
        "mahto2022_firefly_dual",
        "dwt_hd_svd_2025",
    ]


def test_kumar2021_report_similarity_clean_no_attack():
    """Paper reports average PSNR=51.6145 dB, SSIM=0.9992 and NCC=1.

    The local image set is not identical to the paper table, so this test checks the same
    clean/no-attack regime and requires at least paper-level imperceptibility/recovery.
    """
    host, wm = _host_wm()
    row = _metric_row(Kumar2021DWTEntropy(), host, wm)
    assert row["psnr"] >= 51.0
    assert row["ssim"] >= 0.9990
    assert row["ncc"] >= 0.99
    assert row["ber"] == 0.0


def test_gaata2022_dwt_hess_fwa_report_similarity_clean_no_attack():
    """Paper claims high retrieval and improved image quality from FWA-key selection.

    The implementation follows the published DWT/Hessenberg/key path and uses a
    quantization-aware parity digit for the common uint8 benchmark.  It must still
    preserve high PSNR and high clean watermark recovery.
    """
    host, wm = _host_wm()
    row = _metric_row(Gaata2022DWTHessFWA(), host, wm)
    key = row["key"]
    assert key.metadata.embedding_matrix_shape[0] == key.metadata.embedding_matrix_shape[1]
    assert key.metadata.capacity_bits >= wm.size
    assert key.config.block_size == 4
    assert key.config.h_position == (3, 3)
    assert row["psnr"] >= 45.0
    assert row["nc"] >= 0.90
    assert row["ncc"] >= 0.90
    assert row["ber"] <= 0.05


def test_gaata2022_strict_decimal_mode_is_exact_before_uint8_quantization():
    """Strict decimal embedding is validated in the floating transform domain.

    This guards the paper-described decimal-Hessenberg rule separately from the more
    practical uint8 benchmark mode.
    """
    host, wm = _host_wm()
    method = Gaata2022DWTHessFWA(decimal_position=3)
    _, key = method.embed(host, wm)
    extracted_exact = method.extract_exact_float(key)
    assert ber(wm, extracted_exact) == 0.0


def test_mahto2022_report_similarity_clean_no_attack_and_payload_channels():
    """Paper describes three marks embedded in B-R-G channels and high-quality recovery."""
    host, wm = _host_wm()
    method = Mahto2022FireflyDual()
    row = _metric_row(method, host, wm)
    assert row["psnr"] >= 40.0
    assert row["nc"] >= 0.99
    assert row["ncc"] >= 0.99
    assert row["ber"] == 0.0

    payloads = method.extract_payloads(row["watermarked"], row["key"])
    assert payloads["mac"] == method.mac
    assert "AADHAR_DUMMY=" in payloads["payload"]
    assert "BLUE_HASH=" in payloads["payload"]
    # The paper is explicitly multi-channel; all three channels should carry data.
    assert not np.array_equal(row["watermarked"][:, :, 0], host[:, :, 0])
    assert not np.array_equal(row["watermarked"][:, :, 1], host[:, :, 1])
    assert not np.array_equal(row["watermarked"][:, :, 2], host[:, :, 2])


def test_dwt_hd_svd2025_report_similarity_clean_no_attack():
    """Paper reports average PSNR=45.3437 dB, SSIM=0.9987 and average NCC above 0.95."""
    host, wm = _host_wm()
    row = _metric_row(DWTHDSVD2025(), host, wm)
    assert 40.0 <= row["psnr"] <= 50.0
    assert row["ssim"] >= 0.995
    assert row["ncc"] >= 0.95
    assert row["ber"] <= 0.02
