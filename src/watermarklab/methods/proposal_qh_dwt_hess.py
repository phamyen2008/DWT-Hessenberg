from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import math
import random
from typing import Any

import numpy as np
from scipy.linalg import hessenberg

from watermarklab.common.dwt import split_4bands, merge_4bands
from watermarklab.common.chaos import arnold_scramble, arnold_unscramble
from watermarklab.common.metrics import psnr, nc, ber
from watermarklab.common.attack import lite_attack_suite, apply_attack

# Flags follow the original notebook convention.
FLAG_Q4 = 0
FLAG_HPOS = 1
FLAG_SKIP = 2
HPOS_NONE = -1

HPOS_CANDIDATES: tuple[tuple[str, tuple[int, int]], ...] = (
    ("h21", (1, 0)),
    ("h22", (1, 1)),
)

EPS_CONF = 1e-12


@dataclass
class ProposalParams:
    """Parameters of the Q-H DWT-Hessenberg proposal method.

    Defaults are intentionally close to the uploaded notebook, but without the
    notebook's Colab/global-state dependencies.
    """

    arnold_iterations: int = 17
    dwt_bands: tuple[str, ...] = ("LL", "HL", "HH", "LH")
    dwt_mode: str = "orthonormal"  # closest to pywt.dwt2(..., 'haar') for even images
    block_size: int = 4
    private_key: str = "KB123"
    repeat: int | None = 3  # None = full notebook-style structured repetition; integer = practical benchmark speed

    q4_tau: float = 0.50
    q4_margin: float = 0.08
    h01_q: float = 7.0
    h01_margin: float = 0.90

    min_survival_rate: float = 0.75
    bss_weight: float = 0.75
    mse_weight: float = 0.25
    max_q_cand_mse: float = 0.18
    max_h_cand_mse: float = 0.24

    q4_givens_theta_max: float = 0.42
    q4_givens_coarse_steps: int = 13
    q4_givens_fine_theta_max: float = 0.08
    q4_givens_fine_steps: int = 0
    q4_givens_passes: int = 1
    q4_givens_mse_weight: float = 0.10
    q4_givens_extra_margin_weight: float = 0.04
    q4_givens_pairs: tuple[tuple[int, int], ...] = ((0, 2), (1, 3), (0, 1), (1, 2))
    fast_candidate_scoring: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ProposalParams":
        params = cls()
        if not data:
            return params
        for key, value in data.items():
            if not hasattr(params, key):
                continue
            if key in {"dwt_bands", "q4_givens_pairs"}:
                value = tuple(tuple(x) if isinstance(x, list) else x for x in value)
            setattr(params, key, value)
        if isinstance(params.dwt_bands, list):
            params.dwt_bands = tuple(params.dwt_bands)
        return params

    def to_dict(self) -> dict[str, Any]:
        out = dict(self.__dict__)
        out["dwt_bands"] = list(self.dwt_bands)
        out["q4_givens_pairs"] = [list(x) for x in self.q4_givens_pairs]
        return out


@dataclass
class ProposalKey:
    flags: list[int]
    hpos_list: list[int]
    schedule: list[tuple[int, int, str, int, int]]  # bit_index, channel, band, row, col
    wm_shape: tuple[int, int]
    params: ProposalParams
    repeat_factor: int
    usable_blocks: int
    total_blocks: int
    support_counts: list[int] = field(default_factory=list)


def _safe_sign(x: float) -> float:
    return 1.0 if float(x) >= 0.0 else -1.0


def _nearest_nonnegative_mod_value(a: float, target: float, q: float) -> float:
    """Nearest non-negative number v such that v mod q is approximately target.

    This is the QIM step used by the H-domain branch:
      bit 0 -> target q/4, bit 1 -> target 3q/4.
    """

    q = max(float(q), EPS_CONF)
    target = float(target)
    k0 = int(np.round((float(a) - target) / q))
    candidates = []
    for k in range(k0 - 3, k0 + 4):
        v = target + k * q
        if v >= 0:
            candidates.append(v)
    if not candidates:
        return max(0.0, target)
    return float(min(candidates, key=lambda v: abs(v - a)))


def _q4_stat(qm: np.ndarray) -> float:
    """Q4 decision statistic: q22^2 + q32^2 with zero-based [1,1] and [2,1]."""

    return float(qm[1, 1] ** 2 + qm[2, 1] ** 2)


def _givens4(a: int, b: int, theta: float) -> np.ndarray:
    g = np.eye(4, dtype=np.float64)
    c = math.cos(float(theta))
    s = math.sin(float(theta))
    g[a, a] = c
    g[a, b] = s
    g[b, a] = -s
    g[b, b] = c
    return g


def _q4_signed_margin_from_q(qm: np.ndarray, bit: int, params: ProposalParams) -> float:
    d = _q4_stat(qm) - float(params.q4_tau)
    return float(d) if int(bit) == 1 else -float(d)


def _q4_givens_loss(block0: np.ndarray, hh: np.ndarray, qtrial: np.ndarray, bit: int, params: ProposalParams):
    signed_margin = _q4_signed_margin_from_q(qtrial, bit, params)
    target = float(params.q4_margin)
    if signed_margin >= target:
        margin_loss = 0.0
        extra_margin = signed_margin - target
    else:
        margin_loss = target - signed_margin
        extra_margin = 0.0
    block_trial = qtrial @ hh @ qtrial.T
    mse = float(np.mean((block0 - block_trial) ** 2))
    loss = (
        margin_loss
        + float(params.q4_givens_mse_weight) * math.sqrt(max(mse, 0.0))
        + float(params.q4_givens_extra_margin_weight) * max(extra_margin, 0.0)
    )
    return float(loss), float(mse), float(signed_margin)


def _build_q4_candidate(block: np.ndarray, bit: int, params: ProposalParams) -> np.ndarray:
    hh, qm = hessenberg(np.asarray(block, dtype=np.float64), calc_q=True)
    best_q = qm.copy()
    best_loss, best_mse, best_margin = _q4_givens_loss(block, hh, best_q, bit, params)
    if best_margin >= float(params.q4_margin):
        return best_q @ hh @ best_q.T

    grids = [np.linspace(-float(params.q4_givens_theta_max), float(params.q4_givens_theta_max), int(params.q4_givens_coarse_steps))]
    if int(params.q4_givens_fine_steps) > 1:
        grids.append(np.linspace(-float(params.q4_givens_fine_theta_max), float(params.q4_givens_fine_theta_max), int(params.q4_givens_fine_steps)))

    for pass_idx in range(max(1, int(params.q4_givens_passes))):
        theta_grid = grids[min(pass_idx, len(grids) - 1)]
        improved = False
        for a, b in params.q4_givens_pairs:
            local_best_q = best_q
            local_best_key = (best_loss, best_mse, -best_margin)
            for theta in theta_grid:
                if abs(float(theta)) < 1e-15:
                    continue
                qtrial = _givens4(a, b, float(theta)) @ best_q
                loss, mse, signed_margin = _q4_givens_loss(block, hh, qtrial, bit, params)
                key = (loss, mse, -signed_margin)
                if key < local_best_key:
                    local_best_key = key
                    local_best_q = qtrial
            if local_best_q is not best_q:
                best_q = local_best_q
                best_loss, best_mse, best_margin = _q4_givens_loss(block, hh, best_q, bit, params)
                improved = True
        if not improved:
            break
        if best_margin >= float(params.q4_margin):
            break
    return best_q @ hh @ best_q.T


def _build_hpos_candidate(block: np.ndarray, bit: int, params: ProposalParams, pos: tuple[int, int]) -> np.ndarray:
    hh, qm = hessenberg(np.asarray(block, dtype=np.float64), calc_q=True)
    h2 = hh.copy()
    q_h01 = max(float(params.h01_q), EPS_CONF)
    margin = min(max(0.0, float(params.h01_margin)), 0.49 * q_h01)
    rr, cc = pos
    v = float(h2[rr, cc])
    sgn = _safe_sign(v)
    a = abs(v)
    z = a % q_h01
    safe = False
    if int(bit) == 1:
        safe = (0.5 * q_h01 + margin) <= z <= (q_h01 - margin)
    else:
        safe = margin <= z <= (0.5 * q_h01 - margin)
    if not safe:
        target = 0.75 * q_h01 if int(bit) == 1 else 0.25 * q_h01
        h2[rr, cc] = sgn * _nearest_nonnegative_mod_value(a, target, q_h01)
    return qm @ h2 @ qm.T


def _extract_bit_from_candidate_block(block: np.ndarray, mode: int, params: ProposalParams, pos: tuple[int, int] | None = None) -> int:
    hh, qm = hessenberg(np.asarray(block, dtype=np.float64), calc_q=True)
    if int(mode) == FLAG_Q4:
        return 1 if _q4_stat(qm) >= float(params.q4_tau) else 0
    if int(mode) == FLAG_HPOS:
        if pos is None:
            raise ValueError("H-domain candidate requires a coefficient position")
        rr, cc = pos
        z = abs(float(hh[rr, cc])) % max(float(params.h01_q), EPS_CONF)
        return 1 if z >= 0.5 * float(params.h01_q) else 0
    raise ValueError("Unknown candidate mode")


def _candidate_attack_score(block0: np.ndarray, block_cand: np.ndarray, mode: int, bit: int, params: ProposalParams, pos=None):
    mse = float(np.mean((block0 - block_cand) ** 2))
    base = block_cand.astype(np.float64)
    stress_tests = [
        base,
        np.rint(base),
        base + 0.25,
        base - 0.25,
    ]
    if not bool(params.fast_candidate_scoring):
        stress_tests.extend([1.01 * base, 0.99 * base, base + 0.50, base - 0.50])
    passed = 0
    for test_block in stress_tests:
        try:
            passed += int(_extract_bit_from_candidate_block(test_block, mode, params, pos=pos) == int(bit))
        except Exception:
            pass
    survival = float(passed / max(len(stress_tests), 1))
    ok = survival >= float(params.min_survival_rate)
    return survival, mse, survival, survival, survival, ok


def _candidate_mse_limit(cand: dict, params: ProposalParams) -> float:
    if int(cand["flag"]) == FLAG_Q4:
        return max(float(params.max_q_cand_mse), EPS_CONF)
    if int(cand["flag"]) == FLAG_HPOS:
        return max(float(params.max_h_cand_mse), EPS_CONF)
    return 1.0


def _combined_selection_score(cand: dict, params: ProposalParams) -> float:
    bss = float(cand["score"])
    mse = float(cand["mse"])
    normalized_mse = min(mse / _candidate_mse_limit(cand, params), 2.0)
    return float(params.bss_weight) * bss - float(params.mse_weight) * normalized_mse


def _candidate_rank_key(cand: dict, params: ProposalParams):
    return (_combined_selection_score(cand, params), float(cand["score"]), -float(cand["mse"]))


def _candidate_is_strong_enough(cand: dict, params: ProposalParams) -> bool:
    if not bool(cand["ok"]):
        return False
    if float(cand["score"]) < float(params.min_survival_rate):
        return False
    if int(cand["flag"]) == FLAG_Q4 and float(cand["mse"]) > float(params.max_q_cand_mse):
        return False
    if int(cand["flag"]) == FLAG_HPOS and float(cand["mse"]) > float(params.max_h_cand_mse):
        return False
    return True


def _best_hpos_candidate(block0: np.ndarray, bit: int, params: ProposalParams) -> dict:
    all_h: list[dict] = []
    valid_h: list[dict] = []
    for hpos_idx, (hname, pos) in enumerate(HPOS_CANDIDATES):
        block_h = _build_hpos_candidate(block0, bit, params, pos)
        score, mse, raw, avg, worst, ok = _candidate_attack_score(block0, block_h, FLAG_HPOS, bit, params, pos=pos)
        cand = {
            "flag": FLAG_HPOS,
            "block": block_h,
            "score": score,
            "mse": mse,
            "raw": raw,
            "avg": avg,
            "worst": worst,
            "ok": ok,
            "hpos_idx": hpos_idx,
            "hpos_name": hname,
            "pos": pos,
        }
        all_h.append(cand)
        if _candidate_is_strong_enough(cand, params):
            valid_h.append(cand)
    if valid_h:
        return min(valid_h, key=lambda c: float(c["mse"]))
    return min(all_h, key=lambda c: float(c["mse"]))


def _all_block_positions(bands_by_ch: dict[int, dict[str, np.ndarray]], params: ProposalParams) -> list[tuple[int, str, int, int]]:
    positions: list[tuple[int, str, int, int]] = []
    bs = int(params.block_size)
    for ch in sorted(bands_by_ch):
        for band_name in params.dwt_bands:
            arr = bands_by_ch[ch][band_name]
            h0 = (arr.shape[0] // bs) * bs
            w0 = (arr.shape[1] // bs) * bs
            for r in range(0, h0, bs):
                for c in range(0, w0, bs):
                    positions.append((ch, band_name, r, c))
    return positions


def _shuffle_positions(positions: list[tuple[int, str, int, int]], private_key: str) -> list[tuple[int, str, int, int]]:
    key_bytes = hashlib.sha256(str(private_key).encode("utf-8")).digest()
    seed_int = int.from_bytes(key_bytes, "big")
    rng = random.Random(seed_int)
    idx = list(range(len(positions)))
    rng.shuffle(idx)
    return [positions[i] for i in idx]


def _make_structured_schedule(blocks: list[tuple[int, str, int, int]], payload_len: int, repeat: int | None):
    payload_len = int(payload_len)
    max_repeat = len(blocks) // payload_len
    if max_repeat <= 0:
        raise ValueError(f"Capacity insufficient: payload_len={payload_len}, total_blocks={len(blocks)}")
    repeat_factor = max_repeat if repeat is None else max(1, min(int(repeat), max_repeat))
    usable_blocks = repeat_factor * payload_len
    selected = blocks[:usable_blocks]
    schedule: list[tuple[int, int, str, int, int]] = []
    for rep in range(repeat_factor):
        base = rep * payload_len
        for bit_idx in range(payload_len):
            ch, band, r, c = selected[base + bit_idx]
            schedule.append((bit_idx, ch, band, r, c))
    return schedule, repeat_factor, usable_blocks


def _force_binary_64(watermark_binary: np.ndarray) -> np.ndarray:
    wm = (np.asarray(watermark_binary, dtype=np.uint8) >= 127).astype(np.uint8)
    if wm.shape != (64, 64):
        raise ValueError(f"Proposal method expects 64x64 binary watermark, got {wm.shape}")
    return wm


class ProposalQHDWTHess:
    """Faithful Python conversion of the uploaded optimal_dwt_hess notebook.

    Compared with the first cleaned version, this restores the notebook's main
    mathematical components:
      - pywt-compatible orthonormal Haar DWT ordering,
      - Q4 branch using q22^2 + q32^2 thresholding,
      - H-domain branch using h21/h22 QIM,
      - Bit Survival Score candidate screening,
      - structured repetition/majority voting across available color subbands.

    Optimizer is optional. By default it is OFF, so the method is deterministic
    and directly comparable to baselines. Use use_optimizer=True only for an
    adaptive/oracle proposal experiment.
    """

    name = "Proposal_QH_DWT_Hess"

    def __init__(
        self,
        params: ProposalParams | dict[str, Any] | None = None,
        *,
        use_optimizer: bool = False,
        optimizer_trials: int = 4,
        optimizer_seed: int = 123,
    ):
        self.params = params if isinstance(params, ProposalParams) else ProposalParams.from_dict(params)
        self.use_optimizer = bool(use_optimizer)
        self.optimizer_trials = int(optimizer_trials)
        self.optimizer_seed = int(optimizer_seed)

    def _prepare_bands(self, host_rgb: np.ndarray, params: ProposalParams) -> dict[int, dict[str, np.ndarray]]:
        img = np.asarray(host_rgb, dtype=np.uint8)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("Proposal method expects RGB 24-bit image")
        return {ch: split_4bands(img[:, :, ch].astype(np.float64), mode=params.dwt_mode) for ch in range(3)}

    def _embed_with_params(self, host_rgb: np.ndarray, watermark_binary: np.ndarray, params: ProposalParams):
        wm_bits_2d = _force_binary_64(watermark_binary)
        scrambled = arnold_scramble(wm_bits_2d, int(params.arnold_iterations))
        payload_bits = scrambled.reshape(-1).astype(np.uint8)
        bands_by_ch = self._prepare_bands(host_rgb, params)

        positions = _shuffle_positions(_all_block_positions(bands_by_ch, params), params.private_key)
        schedule, repeat_factor, usable_blocks = _make_structured_schedule(positions, payload_bits.size, params.repeat)

        flags: list[int] = []
        hpos_list: list[int] = []
        support_counts = np.zeros(payload_bits.size, dtype=np.int32)
        bs = int(params.block_size)

        for bit_idx, ch, band_name, r, c in schedule:
            bit = int(payload_bits[bit_idx])
            band = bands_by_ch[ch][band_name]
            block0 = band[r : r + bs, c : c + bs].copy()
            try:
                block_q4 = _build_q4_candidate(block0, bit, params)
                score_q4, mse_q4, raw_q4, avg_q4, worst_q4, ok_q4 = _candidate_attack_score(block0, block_q4, FLAG_Q4, bit, params)
                q_cand = {
                    "flag": FLAG_Q4,
                    "block": block_q4,
                    "score": score_q4,
                    "mse": mse_q4,
                    "raw": raw_q4,
                    "avg": avg_q4,
                    "worst": worst_q4,
                    "ok": ok_q4,
                    "hpos_idx": HPOS_NONE,
                    "hpos_name": "none",
                }
                best_h = _best_hpos_candidate(block0, bit, params)
                candidates = [q_cand, best_h]
                strong = [cand for cand in candidates if _candidate_is_strong_enough(cand, params)]
                if strong:
                    best = sorted(strong, key=lambda cand: _candidate_rank_key(cand, params), reverse=True)[0]
                    band[r : r + bs, c : c + bs] = best["block"]
                    flags.append(int(best["flag"]))
                    hpos_list.append(int(best.get("hpos_idx", HPOS_NONE)))
                    support_counts[bit_idx] += 1
                else:
                    flags.append(FLAG_SKIP)
                    hpos_list.append(HPOS_NONE)
            except Exception:
                flags.append(FLAG_SKIP)
                hpos_list.append(HPOS_NONE)

        out = np.asarray(host_rgb, dtype=np.uint8).copy()
        for ch in range(3):
            rec = merge_4bands(bands_by_ch[ch], mode=params.dwt_mode)
            out[:, :, ch] = np.clip(np.rint(rec), 0, 255).astype(np.uint8)

        key = ProposalKey(
            flags=flags,
            hpos_list=hpos_list,
            schedule=schedule,
            wm_shape=wm_bits_2d.shape,
            params=params,
            repeat_factor=repeat_factor,
            usable_blocks=usable_blocks,
            total_blocks=len(positions),
            support_counts=support_counts.astype(int).tolist(),
        )
        return out, key

    def _quick_optimize_params(self, host_rgb: np.ndarray, watermark_binary: np.ndarray) -> ProposalParams:
        """Small optional optimizer for adaptive experiments.

        This is deliberately conservative: it tests a few parameter variants and
        chooses the highest score based on clean PSNR/NC plus a tiny attack subset.
        It is OFF by default to keep fair comparisons reproducible.
        """

        rng = np.random.default_rng(self.optimizer_seed)
        base = self.params
        candidates: list[ProposalParams] = [ProposalParams.from_dict(base.to_dict())]
        band_options = [("LH", "HL"), ("LL", "HL"), ("LL", "LH", "HL"), ("LL", "HL", "HH", "LH")]
        for _ in range(max(0, self.optimizer_trials - 1)):
            p = ProposalParams.from_dict(base.to_dict())
            p.arnold_iterations = int(rng.integers(1, 64))
            p.dwt_bands = tuple(band_options[int(rng.integers(0, len(band_options)))])
            p.h01_q = float(rng.uniform(5.0, 10.0))
            p.h01_margin = min(float(rng.uniform(0.50, 1.20)), 0.49 * p.h01_q)
            p.q4_margin = float(rng.uniform(0.04, 0.14))
            # Keep optimizer fast; final run can still use auto repeat if base uses it.
            p.repeat = base.repeat if base.repeat is not None else min(4, 49152 // 4096)
            candidates.append(p)

        best_score = -1e18
        best_params = base
        wm = _force_binary_64(watermark_binary) * 255
        for p in candidates:
            try:
                watermarked, key = self._embed_with_params(host_rgb, watermark_binary, p)
                extracted = self.extract(watermarked, key, host_rgb=host_rgb)
                clean_psnr = psnr(host_rgb, watermarked)
                clean_nc = nc(wm, extracted)
                clean_ber = ber(wm, extracted)
                attack_ncs = []
                for attack in lite_attack_suite(include_none=False)[:3]:
                    attacked = apply_attack(watermarked, attack)
                    ext_att = self.extract(attacked, key, host_rgb=host_rgb)
                    attack_ncs.append(nc(wm, ext_att))
                mean_attack_nc = float(np.mean(attack_ncs)) if attack_ncs else clean_nc
                # Strong penalty if visibility is too low; otherwise emphasize robustness.
                score = mean_attack_nc + 0.25 * clean_nc - 0.50 * clean_ber + 0.01 * min(clean_psnr, 60.0)
                if score > best_score:
                    best_score = score
                    best_params = p
            except Exception:
                continue
        return best_params

    def embed(self, host_rgb: np.ndarray, watermark_binary: np.ndarray):
        params = self._quick_optimize_params(host_rgb, watermark_binary) if self.use_optimizer else self.params
        return self._embed_with_params(host_rgb, watermark_binary, params)

    def extract(self, possibly_attacked_rgb: np.ndarray, key: ProposalKey, host_rgb: np.ndarray | None = None):
        params = key.params
        img = np.asarray(possibly_attacked_rgb, dtype=np.uint8)
        bands_by_ch = {ch: split_4bands(img[:, :, ch].astype(np.float64), mode=params.dwt_mode) for ch in range(3)}
        num_bits = int(key.wm_shape[0] * key.wm_shape[1])
        votes = [[] for _ in range(num_bits)]
        bs = int(params.block_size)
        n = min(len(key.schedule), len(key.flags), len(key.hpos_list))
        for idx in range(n):
            bit_idx, ch, band_name, r, c = key.schedule[idx]
            flag = int(key.flags[idx])
            if flag == FLAG_SKIP:
                continue
            try:
                block = bands_by_ch[ch][band_name][r : r + bs, c : c + bs]
                if block.shape != (bs, bs):
                    continue
                if flag == FLAG_Q4:
                    extracted_bit = _extract_bit_from_candidate_block(block, FLAG_Q4, params)
                elif flag == FLAG_HPOS:
                    hpos_idx = int(key.hpos_list[idx])
                    if not (0 <= hpos_idx < len(HPOS_CANDIDATES)):
                        continue
                    _, pos = HPOS_CANDIDATES[hpos_idx]
                    extracted_bit = _extract_bit_from_candidate_block(block, FLAG_HPOS, params, pos=pos)
                else:
                    continue
                votes[int(bit_idx)].append(int(extracted_bit))
            except Exception:
                continue
        bits = np.zeros(num_bits, dtype=np.uint8)
        for i, v in enumerate(votes):
            if not v:
                bits[i] = 0
            else:
                ones = int(np.sum(v))
                zeros = len(v) - ones
                bits[i] = 1 if ones >= zeros else 0
        scrambled = bits.reshape(key.wm_shape)
        recovered = arnold_unscramble(scrambled, int(params.arnold_iterations))
        return (recovered * 255).astype(np.uint8)


__all__ = [
    "ProposalQHDWTHess",
    "ProposalParams",
    "ProposalKey",
    "FLAG_Q4",
    "FLAG_HPOS",
    "FLAG_SKIP",
    "HPOS_CANDIDATES",
    "_nearest_nonnegative_mod_value",
    "_q4_stat",
    "_build_hpos_candidate",
    "_extract_bit_from_candidate_block",
]
