# Changelog V3 - paper-compliance and math tests

## New tests

Added a stricter test layer for paper-compliance and shared math correctness.

- `tests/test_common_paper_math.py`
  - YCbCr forward/inverse matrix checks.
  - Haar-DWT known 2x2 formula and energy preservation.
  - LWT/IWT predict-update formulas and round-trip.
  - Visual entropy and edge entropy formulas.
  - Arnold Cat Map one-iteration matrix mapping and inverse.
  - Logistic chaos encryption/decryption round-trip.
  - SVD principal-component and Hessenberg reconstruction tests.
  - Alpha-blending and additive embedding equation tests.

- `tests/test_baseline_paper_compliance.py`
  - Common 512x512 RGB + 64x64 binary watermark contract for all baselines.
  - Roy 2018 32x32 block + three-level DWT + 4x4 SVD contract.
  - Kumar 2021 LWT + visual/edge entropy + Arnold 50 iterations contract.
  - IWT-Hess-SVD 2024 red-channel + 3-level cover IWT + 2-level watermark IWT + PC modification contract.
  - DWT-HD-SVD 2025 Y-channel + Haar-DWT + Hessenberg + SVD + logistic chaos + Eq. (17) contract.
  - Proposal optimizer OFF by default and explicitly enableable.

## Code corrections

- Added `common/entropy.py` for visual entropy, edge entropy, and B_E = E_v - E_e block selection.
- Added `common/embedding_math.py` for alpha-blending/additive equations and inverse extraction formulas.
- Updated Kumar baseline to use LWT/IWT and Arnold Cat Map instead of the earlier simplified DWT-only version.
- Updated DWT-HD-SVD 2025 baseline to use the paper Eq. (17): `S_H' = alpha*S_w + (1-alpha)*S_H`.
- Kept the common-benchmark adaptation for methods with original 256x256 watermark settings.

## Verification

```text
40 passed
```
