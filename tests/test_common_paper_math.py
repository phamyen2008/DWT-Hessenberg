import math
import numpy as np

from watermarklab.common.color import rgb_to_ycbcr, ycbcr_to_rgb
from watermarklab.common.dwt import dwt2_haar_orthonormal, idwt2_haar_orthonormal
from watermarklab.common.iwt import _haar_lift_1d_forward, _haar_lift_1d_inverse, iwt2, iiwt2
from watermarklab.common.entropy import visual_entropy, edge_entropy, block_entropy_score
from watermarklab.common.chaos import arnold_scramble, arnold_unscramble, chaotic_encrypt_uint8, chaotic_decrypt_uint8
from watermarklab.common.embedding_math import (
    additive_embed,
    alpha_blend_cover_weight,
    alpha_blend_payload_weight,
    extract_from_alpha_blend_cover_weight,
    extract_from_alpha_blend_payload_weight,
)
from watermarklab.common.linalg_utils import hess_decompose, hess_reconstruct, svd_pc, diag_from_s


def test_ycbcr_forward_matrix_matches_paper_equation_with_offsets():
    rgb = np.array([[[100, 150, 200]]], dtype=np.uint8)
    y, cb, cr = rgb_to_ycbcr(rgb)
    assert np.isclose(y[0, 0], 0.299 * 100 + 0.587 * 150 + 0.114 * 200, atol=1e-12)
    # The papers round the standard coefficients to three decimals; the implementation uses the standard BT.601 constants.
    assert np.isclose(cb[0, 0], -0.169 * 100 - 0.331 * 150 + 0.500 * 200 + 128, atol=0.08)
    assert np.isclose(cr[0, 0], 0.500 * 100 - 0.419 * 150 - 0.081 * 200 + 128, atol=0.08)


def test_ycbcr_inverse_is_close_after_uint8_rounding():
    rgb = np.array([[[12, 34, 56], [200, 150, 100]]], dtype=np.uint8)
    y, cb, cr = rgb_to_ycbcr(rgb)
    rec = ycbcr_to_rgb(y, cb, cr)
    assert np.max(np.abs(rec.astype(int) - rgb.astype(int))) <= 1


def test_orthonormal_haar_dwt_known_2x2_formula_and_inverse():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    ll, lh, hl, hh = dwt2_haar_orthonormal(x)
    assert np.allclose(ll, [[5.0]])
    assert np.allclose(lh, [[-2.0]])
    assert np.allclose(hl, [[-1.0]])
    assert np.allclose(hh, [[0.0]])
    assert np.allclose(idwt2_haar_orthonormal(ll, lh, hl, hh), x)


def test_haar_orthonormal_energy_preservation():
    rng = np.random.default_rng(100)
    x = rng.normal(size=(16, 16))
    bands = dwt2_haar_orthonormal(x)
    energy_bands = sum(float(np.sum(b ** 2)) for b in bands)
    assert np.isclose(float(np.sum(x ** 2)), energy_bands, atol=1e-9)


def test_lifting_wavelet_predict_update_1d_formula():
    x = np.array([[2.0, 5.0, 4.0, 9.0]])
    low, high = _haar_lift_1d_forward(x, axis=1)
    assert np.allclose(high, [[3.0, 5.0]])  # odd - even
    assert np.allclose(low, [[3.5, 6.5]])   # even + high/2
    assert np.allclose(_haar_lift_1d_inverse(low, high, axis=1), x)


def test_lifting_wavelet_2d_roundtrip_integer_values():
    x = np.arange(64, dtype=np.float64).reshape(8, 8)
    assert np.allclose(iiwt2(*iwt2(x)), x)


def test_visual_and_edge_entropy_match_paper_definitions():
    block = np.array([[0, 1], [0, 1]], dtype=np.uint8)
    ev = visual_entropy(block, bins=2)
    ee = edge_entropy(block, bins=2)
    assert np.isclose(ev, 1.0, atol=1e-12)  # -2*(0.5 log2 0.5)
    assert np.isclose(ee, math.exp(0.5), atol=1e-12)  # sum_i p_i e^(1-p_i)
    assert np.isclose(block_entropy_score(block, bins=2), ev - ee, atol=1e-12)


def test_arnold_cat_map_one_iteration_matches_matrix_mod_n():
    x = np.arange(16, dtype=np.uint8).reshape(4, 4)
    scrambled = arnold_scramble(x, iterations=1)
    expected = np.zeros_like(x)
    n = 4
    for r in range(n):
        for c in range(n):
            expected[(r + c) % n, (r + 2 * c) % n] = x[r, c]
    assert np.array_equal(scrambled, expected)
    assert np.array_equal(arnold_unscramble(scrambled, iterations=1), x)


def test_logistic_chaotic_encrypt_decrypt_is_exact_permutation_xor_roundtrip():
    mat = np.arange(64, dtype=np.uint8).reshape(8, 8)
    encrypted, idx, mask = chaotic_encrypt_uint8(mat, x0=0.5, mu=4.0)
    assert encrypted.shape == mat.shape
    assert sorted(idx.tolist()) == list(range(mat.size))
    assert mask.shape == (mat.size,)
    decrypted = chaotic_decrypt_uint8(encrypted, idx, mask)
    assert np.array_equal(decrypted.astype(np.uint8), mat)


def test_svd_pc_and_hessenberg_math_roundtrips():
    rng = np.random.default_rng(101)
    mat = rng.normal(size=(6, 6))
    pc, vt = svd_pc(mat)
    assert np.allclose(pc @ vt, mat)
    q, h = hess_decompose(mat)
    assert np.allclose(q @ q.T, np.eye(6), atol=1e-10)
    assert np.allclose(hess_reconstruct(q, h), mat, atol=1e-10)
    assert np.allclose(np.tril(h, -2), 0.0, atol=1e-10)


def test_embedding_equations_and_inverse_formulas_are_exact():
    cover = np.array([10.0, 20.0, 30.0])
    payload = np.array([1.0, 2.0, 3.0])
    alpha = 0.25
    assert np.allclose(additive_embed(cover, payload, alpha), cover + alpha * payload)
    marked_cover_weight = alpha_blend_cover_weight(cover, payload, alpha)
    assert np.allclose(marked_cover_weight, alpha * cover + (1 - alpha) * payload)
    assert np.allclose(extract_from_alpha_blend_cover_weight(marked_cover_weight, cover, alpha), payload)
    marked_payload_weight = alpha_blend_payload_weight(cover, payload, alpha)
    assert np.allclose(marked_payload_weight, (1 - alpha) * cover + alpha * payload)
    assert np.allclose(extract_from_alpha_blend_payload_weight(marked_payload_weight, cover, alpha), payload)
