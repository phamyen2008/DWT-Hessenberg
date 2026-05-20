# Baseline validation v5

This release integrates the four uploaded baseline papers into the common benchmark package.

## Added method wrappers

| Method id | Wrapper file | Vendor source |
|---|---|---|
| `gaata2022_dwt_hess_fwa` | `src/watermarklab/methods/gaata2022_dwt_hess_fwa.py` | `src/watermarklab/vendor/dwt_hess_fwa/` |
| `mahto2022_firefly_dual` | `src/watermarklab/methods/mahto2022_firefly_dual.py` | `src/watermarklab/vendor/mahto2022/` |
| `kumar2021` | `src/watermarklab/methods/kumar2021_dwt_entropy.py` | native implementation |
| `dwt_hd_svd_2025` | `src/watermarklab/methods/dwt_hd_svd2025.py` | native implementation |

## New validation tests

`tests/test_uploaded_baseline_paper_report_similarity.py` adds six checks:

1. all four uploaded baselines are registered in `build_methods`,
2. Kumar 2021 clean result is on the same/no-worse scale as the reported PSNR/SSIM/NCC,
3. Gaata 2022 DWT/Hessenberg/FWA path has enough capacity and clean recovery,
4. Gaata 2022 strict decimal mode is exact before uint8 quantization,
5. Mahto 2022 recovers the image watermark and the R/G payloads,
6. DWT-HD-SVD 2025 matches the reported PSNR/NCC scale.

## Why Gaata 2022 has two validation modes

The paper describes embedding by modifying a selected decimal digit of a Hessenberg coefficient.  A strict decimal digit is exact in floating-point extraction but fragile after normal `uint8` image saving.  Therefore:

- strict mode (`decimal_position=3`) is validated with exact floating extraction;
- common benchmark mode (`decimal_position=-1`) is a quantization-aware version of the same parity rule and is used for normal uint8 clean extraction.

## Command used for package verification

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src pytest -q
```

Checked result:

```text
52 passed, 1 skipped
```

Clean four-baseline smoke command:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=src python main.py \
  --methods kumar2021,gaata2022_dwt_hess_fwa,mahto2022_firefly_dual,dwt_hd_svd_2025 \
  --max-images 1 --no-save-images --attack-preset none \
  --output results/four_baseline_clean_smoke
```
