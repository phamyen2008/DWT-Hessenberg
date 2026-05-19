from pathlib import Path
import numpy as np
from watermarklab.common.io_utils import load_host_rgb, load_watermark_binary
from watermarklab.common.metrics import psnr, nc, ncc, ber
from watermarklab.methods import build_methods

ROOT = Path(__file__).resolve().parents[1]


def test_each_method_runs_no_attack_on_real_input_quick():
    host = load_host_rgb(ROOT / "data" / "host" / "lenna.bmp")
    wm = load_watermark_binary(ROOT / "data" / "watermark" / "wm.png")
    # Proposal is tested with repeat=1 here so the unit test validates the math
    # path without making pytest slow. Use --proposal-repeat auto in main.py for
    # the notebook-faithful/high-robustness benchmark.
    methods = build_methods(["kumar2021", "roy2018", "iwt_hess_svd_2024", "dwt_hd_svd_2025", "proposal"], proposal_options={"params": {"repeat": 1}})
    for method_id, method in methods.items():
        watermarked, key = method.embed(host, wm)
        extracted = method.extract(watermarked, key, host_rgb=host)
        assert watermarked.shape == host.shape
        assert watermarked.dtype == np.uint8
        assert extracted.shape == wm.shape
        assert extracted.dtype == np.uint8
        assert psnr(host, watermarked) > 20.0
        assert 0.0 <= nc(wm, extracted) <= 1.0000001
        assert -1.0000001 <= ncc(wm, extracted) <= 1.0000001
        assert 0.0 <= ber(wm, extracted) <= 1.0
