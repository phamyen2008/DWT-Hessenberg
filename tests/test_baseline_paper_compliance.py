from pathlib import Path
import numpy as np

from watermarklab.common.io_utils import load_host_rgb, load_watermark_binary
from watermarklab.common.color import rgb_to_ycbcr
from watermarklab.common.iwt import iwt2
from watermarklab.common.dwt import multilevel_dwt_ll
from watermarklab.common.chaos import arnold_scramble
from watermarklab.common.entropy import select_max_entropy_score_block
from watermarklab.common.metrics import psnr, nc, ncc, ber
from watermarklab.methods import build_methods
from watermarklab.methods.kumar2021_dwt_entropy import Kumar2021DWTEntropy
from watermarklab.methods.roy2018_dwt_svd import Roy2018DWTSVD
from watermarklab.methods.iwt_hess_svd_2024 import IWTHessSVD2024
from watermarklab.methods.dwt_hd_svd2025 import DWTHDSVD2025
from watermarklab.methods.proposal_qh_dwt_hess import ProposalQHDWTHess, ProposalParams

ROOT = Path(__file__).resolve().parents[1]


def _host_wm():
    return (
        load_host_rgb(ROOT / "data" / "host" / "lenna.bmp"),
        load_watermark_binary(ROOT / "data" / "watermark" / "wm.png"),
    )


def test_all_baseline_methods_accept_common_512_rgb_and_64_binary_contract():
    host, wm = _host_wm()
    methods = build_methods(["kumar2021", "roy2018", "iwt_hess_svd_2024", "dwt_hd_svd_2025"])
    for method_id, method in methods.items():
        watermarked, key = method.embed(host, wm)
        extracted = method.extract(watermarked, key, host_rgb=host)
        assert watermarked.shape == (512, 512, 3)
        assert watermarked.dtype == np.uint8
        assert extracted.shape == (64, 64)
        assert extracted.dtype == np.uint8
        assert set(np.unique(extracted)).issubset({0, 255})
        assert psnr(host, watermarked) > 35.0, method_id
        # No-attack extraction should be very close even after RGB/YCbCr rounding.
        assert nc(wm, extracted) >= 0.95, method_id
        assert ncc(wm, extracted) >= 0.90, method_id
        assert ber(wm, extracted) <= 0.05, method_id


def test_kumar2021_matches_lwt_entropy_acm_alpha_blending_contract():
    host, wm = _host_wm()
    method = Kumar2021DWTEntropy()
    watermarked, key = method.embed(host, wm)
    extracted = method.extract(watermarked, key, host_rgb=host)

    assert method.name == "Kumar2021_LWT_Entropy"
    assert key.block_size == 32
    assert key.arnold_iterations == 50
    assert key.watermark_ll.shape == (32, 32)
    assert key.watermark_lh.shape == (32, 32)
    assert key.watermark_hl.shape == (32, 32)

    y, _, _ = rgb_to_ycbcr(host)
    _, _, _, hh = iwt2(y)
    expected_block, _ = select_max_entropy_score_block(hh, block_size=32)
    assert key.block_index == expected_block

    scrambled = arnold_scramble(wm, iterations=50)
    _, _, _, w_hh = iwt2(scrambled.astype(np.float64))
    assert w_hh.shape == (32, 32)
    assert ber(wm, extracted) == 0.0


def test_roy2018_matches_ycbcr_32block_3level_dwt_svd_contract():
    host, wm = _host_wm()
    method = Roy2018DWTSVD()
    watermarked, key = method.embed(host, wm)
    extracted = method.extract(watermarked, key, host_rgb=host)

    assert method.alpha == 0.02
    assert len(key.original_s) == 256  # 512x512 image / 32x32 cover blocks
    assert len(key.uw) == 256
    assert len(key.vwt) == 256
    assert all(s.shape == (4, 4) for s in key.original_s)
    assert all(u.shape == (4, 4) for u in key.uw)
    assert all(v.shape == (4, 4) for v in key.vwt)

    y, _, _ = rgb_to_ycbcr(host)
    first_block = y[:32, :32]
    ll, _ = multilevel_dwt_ll(first_block, levels=3, mode="average")
    assert ll.shape == (4, 4)
    assert ber(wm, extracted) == 0.0


def test_iwt_hess_svd_2024_matches_red_channel_3level_iwt_and_2level_watermark_contract():
    host, wm = _host_wm()
    method = IWTHessSVD2024()
    watermarked, key = method.embed(host, wm)
    extracted = method.extract(watermarked, key, host_rgb=host)

    assert method.alpha == 0.015
    assert key.wm_shape_large == (256, 256)
    assert key.wm_shape_out == (64, 64)
    assert len(key.cover_levels) == 3
    assert len(key.wm_levels) == 2
    assert key.pc_cover.shape == (64, 64)
    assert key.vt_cover.shape == (64, 64)
    assert key.q_cover.shape == (64, 64)
    # Paper embeds in red channel; green and blue should remain exactly unchanged.
    assert not np.array_equal(watermarked[:, :, 0], host[:, :, 0])
    assert np.array_equal(watermarked[:, :, 1], host[:, :, 1])
    assert np.array_equal(watermarked[:, :, 2], host[:, :, 2])
    assert ber(wm, extracted) <= 0.05


def test_dwt_hd_svd_2025_matches_y_channel_haar_hessenberg_svd_logistic_contract():
    host, wm = _host_wm()
    method = DWTHDSVD2025()
    watermarked, key = method.embed(host, wm)
    extracted = method.extract(watermarked, key, host_rgb=host)

    assert method.alpha == 0.08
    assert key.ll_roi_shape == (64, 64)  # common-benchmark adaptation of original 256x256 paper watermark
    assert key.wm_shape_out == (64, 64)
    assert key.uw.shape == (64, 64)
    assert key.vtw.shape == (64, 64)
    assert key.sh_original.shape == (64,)
    assert key.chaotic_indices.shape == (4096,)
    assert key.chaotic_mask.shape == (4096,)
    assert ber(wm, extracted) <= 0.01


def test_proposal_default_does_not_run_optimizer_but_can_be_enabled_explicitly():
    default_method = ProposalQHDWTHess(params=ProposalParams(repeat=1))
    optimized_method = ProposalQHDWTHess(params=ProposalParams(repeat=1), use_optimizer=True, optimizer_trials=2)
    assert default_method.use_optimizer is False
    assert optimized_method.use_optimizer is True
    assert optimized_method.optimizer_trials == 2
