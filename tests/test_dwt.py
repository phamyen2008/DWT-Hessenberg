import numpy as np
from watermarklab.common.dwt import dwt2, idwt2, split_4bands, merge_4bands
from watermarklab.common.iwt import iwt2, iiwt2, multilevel_iwt, multilevel_iiwt


def test_average_haar_reconstructs():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(32, 32))
    ll, lh, hl, hh = dwt2(x, mode="average")
    y = idwt2(ll, lh, hl, hh, mode="average")
    assert np.allclose(x, y, atol=1e-10)


def test_orthonormal_haar_reconstructs_and_preserves_energy():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(32, 32))
    ll, lh, hl, hh = dwt2(x, mode="orthonormal")
    y = idwt2(ll, lh, hl, hh, mode="orthonormal")
    energy_in = np.sum(x * x)
    energy_subbands = sum(float(np.sum(a * a)) for a in [ll, lh, hl, hh])
    assert np.allclose(x, y, atol=1e-10)
    assert np.allclose(energy_in, energy_subbands, atol=1e-9)


def test_split_merge_4bands_matches_dwt_idwt():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(32, 32))
    bands = split_4bands(x, mode="orthonormal")
    y = merge_4bands(bands, mode="orthonormal")
    assert set(bands) == {"LL", "LH", "HL", "HH"}
    assert np.allclose(x, y, atol=1e-10)


def test_iwt_reconstructs():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(32, 32))
    ll, lh, hl, hh = iwt2(x)
    y = iiwt2(ll, lh, hl, hh)
    assert np.allclose(x, y, atol=1e-10)


def test_multilevel_iwt_reconstructs():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(64, 64))
    ll, levels = multilevel_iwt(x, levels=3)
    y = multilevel_iiwt(ll, levels)
    assert np.allclose(x, y, atol=1e-10)
