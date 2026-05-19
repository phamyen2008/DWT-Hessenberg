# Clean Watermarking Project v2

This project refactors the uploaded watermarking code into a clean Python package for a proposal-paper benchmark.

## What changed in v2

- Restored the proposal method closer to `optimal_dwt_hess.ipynb`:
  - pywt-compatible orthonormal Haar DWT math,
  - Q4 branch: `q22^2 + q32^2 >= tau`,
  - H-domain branch: QIM on `h21` / `h22`,
  - Bit Survival Score candidate screening,
  - structured repetition and majority voting,
  - optional optimizer **OFF by default**.
- Added a larger deterministic attack suite in `src/watermarklab/common/attack.py`.
- Added math-focused tests for DWT, IWT, Arnold, chaotic encryption/decryption, SVD, Hessenberg, metrics, and proposal Q/H embedding logic.
- All methods accept the common input format:
  - host image: `512 x 512` RGB / 24-bit color,
  - watermark: `64 x 64` binary.

## Install

```bash
pip install -r requirements.txt
```

## Run tests

```bash
PYTHONPATH=src pytest -q
```

Expected result in the checked package:

```text
23 passed
```

## Run all methods

Practical run, using the lite attack suite and proposal repetition factor 3:

```bash
PYTHONPATH=src python main.py \
  --host-dir data/host \
  --watermark data/watermark/wm.png \
  --output results/common_benchmark \
  --attack-preset lite \
  --proposal-repeat 3
```

For a quick smoke run on one image:

```bash
PYTHONPATH=src python main.py \
  --max-images 1 \
  --no-save-images \
  --attack-preset lite \
  --proposal-repeat 3 \
  --output results/quick_all_methods
```

## Proposal method options

Optimizer is disabled by default for fair comparison. To enable adaptive/oracle proposal optimization:

```bash
PYTHONPATH=src python main.py \
  --methods proposal \
  --proposal-use-optimizer \
  --proposal-optimizer-trials 4 \
  --proposal-repeat 3 \
  --max-images 1 \
  --output results/proposal_optimizer_demo
```

To run the notebook-style full structured repetition, use:

```bash
--proposal-repeat auto
```

`auto` uses all available DWT blocks and is much slower. Use it for final proposal-only experiments, not for every quick test.

## Full attack suite

The full attack suite includes compression, noise, filtering, geometric, occlusion, photometric, and quantization attacks:

```bash
PYTHONPATH=src python main.py \
  --methods proposal \
  --max-images 1 \
  --attack-preset full \
  --proposal-repeat 3 \
  --output results/proposal_full_attack_demo
```

## Main output files

After running `main.py`, the benchmark exports:

```text
per_image_attack_results.csv
summary_by_method_phase.csv
compare_psnr_nc_ber_ncc_before_after_attack.csv
failures.json
```

## Important research note

Use `proposal-repeat 3` for practical fair comparison. Use `proposal-repeat auto` only when you want the closest notebook-style structured repetition, because it can be much slower. If optimizer is enabled, report it separately as an adaptive/oracle variant, not mixed with fixed-parameter baselines.

## V3 paper-compliance tests

This release adds stricter paper-compliance and math-correctness tests. See:

- `docs/PAPER_COMPLIANCE_TEST_PLAN.md`
- `CHANGELOG_REFACTOR_V3.md`

Run:

```bash
PYTHONPATH=src pytest -q
```

Expected in this release:

```text
40 passed
```

## V4: Proposal validation layer

To check that the proposal implementation still follows the notebook-level math contract:

```bash
PYTHONPATH=src python tools/validate_proposal.py --no-end-to-end
```

To include a real no-attack embed/extract check:

```bash
PYTHONPATH=src python tools/validate_proposal.py \
  --host data/host/lenna.bmp \
  --watermark data/watermark/wm.png
```

The validation report is written to:

```text
results/proposal_validation_report.json
```

The attack module now has three presets:

```bash
--attack-preset lite
--attack-preset full
--attack-preset stress
```

`stress` is a compact but harder suite for proposal robustness checks.

To run the slow real proposal end-to-end pytest check:

```bash
RUN_SLOW_PROPOSAL=1 PYTHONPATH=src pytest -q \
  tests/test_proposal_validation_layer.py::test_proposal_end_to_end_validation_layer_on_real_input
```
