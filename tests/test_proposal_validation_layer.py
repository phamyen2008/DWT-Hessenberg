from pathlib import Path
import os
import pytest

from watermarklab.common.io_utils import load_host_rgb, load_watermark_binary
from watermarklab.methods.proposal_qh_dwt_hess import ProposalParams, ProposalQHDWTHess
from watermarklab.proposal_validation import (
    proposal_notebook_contract,
    validate_proposal_params,
    validate_proposal_math_branches,
    validate_structured_schedule,
    validate_proposal_end_to_end,
)

ROOT = Path(__file__).resolve().parents[1]


def test_proposal_contract_matches_notebook_constants():
    c = proposal_notebook_contract()
    assert c["host_shape"] == (512, 512, 3)
    assert c["watermark_shape"] == (64, 64)
    assert c["block_size"] == 4
    assert c["private_key"] == "KB123"
    assert c["arnold_iterations"] == 17
    assert c["dwt_bands"] == ("LL", "HL", "HH", "LH")
    assert c["hpos_candidates"] == (("h21", (1, 0)), ("h22", (1, 1)))


def test_proposal_default_optimizer_is_off_and_params_match_contract():
    method = ProposalQHDWTHess()
    assert method.use_optimizer is False
    report = validate_proposal_params()
    assert report["ok"], report


def test_proposal_branch_math_layer_passes():
    report = validate_proposal_math_branches(ProposalParams(repeat=1))
    assert report["ok"], report


def test_proposal_structured_schedule_uses_full_capacity_correctly():
    auto_report = validate_structured_schedule(total_blocks=49152, payload_len=4096, repeat=None)
    assert auto_report["ok"], auto_report
    assert auto_report["repeat_factor"] == 12
    fixed_report = validate_structured_schedule(total_blocks=49152, payload_len=4096, repeat=3)
    assert fixed_report["ok"], fixed_report
    assert fixed_report["repeat_factor"] == 3
    assert fixed_report["usable_blocks"] == 3 * 4096


@pytest.mark.skipif(os.getenv("RUN_SLOW_PROPOSAL") != "1", reason="Set RUN_SLOW_PROPOSAL=1 to run real proposal end-to-end validation")
def test_proposal_end_to_end_validation_layer_on_real_input():
    host = load_host_rgb(ROOT / "data" / "host" / "lenna.bmp")
    wm = load_watermark_binary(ROOT / "data" / "watermark" / "wm.png")
    # One repetition keeps pytest practical while still checking real 512x512 RGB + 64x64 WM.
    report = validate_proposal_end_to_end(host, wm, ProposalParams(repeat=1, dwt_bands=("HL",), q4_givens_coarse_steps=3))
    assert report["ok"], report
