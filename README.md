# Clean Watermarking Project v5

This project is a reproducible Python benchmark for color-image watermarking experiments.  It now contains the proposal method plus the four paper-guided baselines supplied with the uploaded reports.

## Common benchmark contract

All benchmark methods accept the same input format:

```text
Host image: 512 x 512 RGB / 24-bit color
Watermark:  64 x 64 binary image
Metrics:    PSNR, SSIM, NC, NCC, BER
```

## Included methods

| Method id | Paper-guided method | Notes |
|---|---|---|
| `kumar2021` | Kumar & Singh 2021 DWT/LWT maximum-entropy baseline | YCbCr-Y channel, DWT/IWT, max-entropy 32x32 block, alpha blending |
| `gaata2022_dwt_hess_fwa` | Gaata et al. 2022 DWT + Hessenberg + Firework Algorithm baseline | RGB split, DWT detail-band embedding matrix, chaotic keys, Hessenberg parity embedding; default common mode is quantization-aware |
| `mahto2022_firefly_dual` | Mahto & Singh 2022 firefly-optimized dual/multi watermark baseline | R-channel text mark, G-channel payload mark, B-channel encrypted image mark |
| `dwt_hd_svd_2025` | DWT-HD-SVD chaotic mapping baseline | YCbCr-Y, DWT LL, Hessenberg/HD, SVD, logistic chaotic watermark encryption |
| `proposal` | Proposed Q/H DWT-Hessenberg method | Optional optimizer is OFF by default |
| `roy2018`, `iwt_hess_svd_2024` | Older internal comparison baselines | Kept for backward compatibility |

## Install

```bash
pip install -r requirements.txt
```

## Run tests

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src pytest -q
```

Expected result in this checked package:

```text
52 passed, 1 skipped
```

The skipped test is the intentionally slow proposal end-to-end validation.  To run it:

```bash
RUN_SLOW_PROPOSAL=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src pytest -q \
  tests/test_proposal_validation_layer.py::test_proposal_end_to_end_validation_layer_on_real_input
```

## Run the four uploaded baselines only

Clean/no-attack smoke test:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python main.py \
  --methods kumar2021,gaata2022_dwt_hess_fwa,mahto2022_firefly_dual,dwt_hd_svd_2025 \
  --max-images 1 \
  --no-save-images \
  --attack-preset none \
  --output results/four_baseline_clean_smoke
```

Lite attack run for a single baseline:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python main.py \
  --methods kumar2021 \
  --max-images 1 \
  --no-save-images \
  --attack-preset lite \
  --output results/kumar2021_lite_smoke
```

Available attack presets:

```text
none    clean extraction only
lite    JPEG/noise/filter/geometric/photometric quick suite
stress  compact but harsher suite
full    large attack suite
```

## Paper-similarity tests for the four baselines

The new test file is:

```text
tests/test_uploaded_baseline_paper_report_similarity.py
```

It checks that the four supplied baselines are registered and that clean/no-attack results match the scale reported in the supplied papers:

- Kumar 2021: PSNR around or above the paper-reported 51.6145 dB, SSIM near 0.999, NCC near 1.
- Gaata 2022: high PSNR and high clean retrieval using the DWT/Hessenberg/key path.
- Mahto 2022: three-channel payload structure and clean watermark recovery.
- DWT-HD-SVD 2025: PSNR close to the reported 45.3437 dB scale and NCC above 0.95.

## Important reproducibility notes

These implementations are **paper-guided reproductions**, not byte-identical reproductions of private author code.  Several papers omit implementation-level choices such as exact wavelet library, contourlet implementation, optimizer iteration count, chaotic-map safeguards, image resizing policy, and side information.  The code documents those assumptions in wrapper classes, tests, and `docs/BASELINE_VALIDATION_V5.md`.

For final manuscript tables, keep these categories separate:

```text
1. Original-paper reported results
2. Paper-guided reproduction results
3. Unified common-benchmark results
4. Proposal-only validation results
```

## Main output files

After running `main.py`, the benchmark exports:

```text
per_image_attack_results.csv
summary_by_method_phase.csv
compare_psnr_nc_ber_ncc_before_after_attack.csv
failures.json
```

## Proposal options

Optimizer is disabled by default for fair comparison. To enable adaptive/oracle proposal optimization:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python main.py \
  --methods proposal \
  --proposal-use-optimizer \
  --proposal-optimizer-trials 4 \
  --proposal-repeat 3 \
  --max-images 1 \
  --output results/proposal_optimizer_demo
```

Use `--proposal-repeat auto` only for proposal-only final experiments because it is much slower.
