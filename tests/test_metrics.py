import math
import numpy as np
from watermarklab.common.metrics import mse, mae, psnr, nc, ncc, ber, to_bits


def test_metrics_on_identical_images_and_watermarks():
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    wm = np.zeros((8, 8), dtype=np.uint8)
    wm[::2, ::2] = 255
    assert psnr(img, img) == float("inf")
    assert nc(wm, wm) == 1.0
    assert ncc(wm, wm) == 1.0
    assert ber(wm, wm) == 0.0


def test_mse_mae_psnr_known_values():
    a = np.array([0, 10, 20], dtype=np.uint8)
    b = np.array([0, 20, 40], dtype=np.uint8)
    assert mse(a, b) == (0**2 + 10**2 + 20**2) / 3
    assert mae(a, b) == 10.0
    expected_psnr = 10.0 * math.log10((255.0**2) / mse(a, b))
    assert abs(psnr(a, b) - expected_psnr) < 1e-12


def test_binary_nc_ncc_ber_known_values():
    a = np.array([[255, 0], [255, 0]], dtype=np.uint8)
    b = np.array([[255, 0], [0, 255]], dtype=np.uint8)
    assert np.array_equal(to_bits(a), np.array([1, 0, 1, 0], dtype=np.uint8))
    assert abs(nc(a, b) - 0.5) < 1e-12
    assert abs(ncc(a, b) - 0.0) < 1e-12
    assert abs(ber(a, b) - 0.5) < 1e-12
