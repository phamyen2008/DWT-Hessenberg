import numpy as np
from scipy.linalg import hessenberg

from watermarklab.common.chaos import (
    logistic_sequence,
    logistic_permutation,
    chaotic_encrypt_uint8,
    chaotic_decrypt_uint8,
    arnold_scramble,
    arnold_unscramble,
)
from watermarklab.common.linalg_utils import diag_from_s, svd_pc, hess_decompose, hess_reconstruct
from watermarklab.methods.proposal_qh_dwt_hess import (
    ProposalParams,
    FLAG_HPOS,
    _nearest_nonnegative_mod_value,
    _q4_stat,
    _build_hpos_candidate,
    _extract_bit_from_candidate_block,
)


def test_logistic_sequence_range_and_permutation_validity():
    seq = logistic_sequence(100, x0=0.321)
    assert seq.shape == (100,)
    assert np.all((seq >= 0.0) & (seq <= 1.0))
    perm = logistic_permutation(100, x0=0.321)
    assert sorted(perm.tolist()) == list(range(100))


def test_chaotic_encrypt_decrypt_roundtrip():
    rng = np.random.default_rng(10)
    mat = rng.integers(0, 256, size=(16, 16), dtype=np.uint8)
    encrypted, idx, mask = chaotic_encrypt_uint8(mat, x0=0.517)
    decrypted = chaotic_decrypt_uint8(encrypted, idx, mask)
    assert np.array_equal(mat.astype(np.float64), decrypted)


def test_arnold_roundtrip_binary_square():
    rng = np.random.default_rng(11)
    bits = rng.integers(0, 2, size=(32, 32), dtype=np.uint8)
    scrambled = arnold_scramble(bits, iterations=17)
    recovered = arnold_unscramble(scrambled, iterations=17)
    assert np.array_equal(bits, recovered)


def test_svd_pc_reconstructs_original_matrix():
    rng = np.random.default_rng(12)
    mat = rng.normal(size=(8, 6))
    pc, vt = svd_pc(mat)
    assert pc.shape == (8, 6)
    assert vt.shape == (6, 6)
    assert np.allclose(pc @ vt, mat)


def test_hessenberg_decompose_reconstructs_original_matrix():
    rng = np.random.default_rng(13)
    mat = rng.normal(size=(4, 4))
    q, h = hess_decompose(mat)
    assert np.allclose(q @ q.T, np.eye(4), atol=1e-10)
    assert np.allclose(hess_reconstruct(q, h), mat, atol=1e-10)


def test_diag_from_s_places_singular_values_on_diagonal():
    s = np.array([3.0, 2.0, 1.0])
    d = diag_from_s(s, shape=(4, 5))
    assert d.shape == (4, 5)
    assert np.array_equal(np.diag(d[:3, :3]), s)
    assert np.count_nonzero(d) == 3


def test_nearest_nonnegative_mod_value_has_correct_residue():
    q = 7.0
    target = 0.75 * q
    value = _nearest_nonnegative_mod_value(22.3, target, q)
    assert value >= 0
    assert abs((value % q) - target) < 1e-12


def test_proposal_hpos_candidate_embeds_requested_bit():
    params = ProposalParams(repeat=1, h01_q=7.0, h01_margin=0.9)
    rng = np.random.default_rng(14)
    block = rng.normal(loc=100.0, scale=10.0, size=(4, 4))
    for bit in [0, 1]:
        candidate = _build_hpos_candidate(block, bit, params, pos=(1, 0))
        extracted = _extract_bit_from_candidate_block(candidate, FLAG_HPOS, params, pos=(1, 0))
        assert extracted == bit


def test_q4_stat_matches_definition():
    _, q = hessenberg(np.arange(16, dtype=np.float64).reshape(4, 4), calc_q=True)
    assert _q4_stat(q) == float(q[1, 1] ** 2 + q[2, 1] ** 2)
