from pathlib import Path
import numpy as np
from watermarklab.common.io_utils import load_host_rgb, load_watermark_binary, list_image_files

ROOT = Path(__file__).resolve().parents[1]


def test_real_inputs_have_expected_shape():
    hosts = list_image_files(ROOT / "data" / "host")
    assert len(hosts) >= 1
    for p in hosts:
        img = load_host_rgb(p)
        assert img.shape == (512, 512, 3)
        assert img.dtype == np.uint8
    wm = load_watermark_binary(ROOT / "data" / "watermark" / "wm.png")
    assert wm.shape == (64, 64)
    assert wm.dtype == np.uint8
    assert set(np.unique(wm).tolist()).issubset({0, 255})
